
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


cpp_fused_native_layer_norm_backward_select_backward_slice_backward_sum_0 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp0 = static_cast<int>(0);
                    auto tmp1 = tmp0 == tmp0;
                    auto tmp3 = static_cast<float>(0.0);
                    auto tmp4 = to_float_mask(tmp1);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = decltype(tmp2)::blendv(tmp5, tmp2, tmp4);
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 * tmp9;
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                    auto tmp0 = static_cast<int>(0);
                    auto tmp1 = tmp0 == tmp0;
                    auto tmp3 = static_cast<float>(0.0);
                    auto tmp4 = to_float_mask(tmp1);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = decltype(tmp2)::blendv(tmp5, tmp2, tmp4);
                    auto tmp8 = tmp6 * tmp7;
                    tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(65L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr4[static_cast<long>(x0)];
                            auto tmp5 = c10::convert<int>(x1);
                            auto tmp6 = static_cast<int>(0);
                            auto tmp7 = tmp5 == tmp6;
                            auto tmp8 = in_ptr1[static_cast<long>(x2 + (1024L*x0))];
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = tmp7 ? tmp8 : tmp9;
                            auto tmp11 = in_ptr2[static_cast<long>(x2)];
                            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                            auto tmp13 = static_cast<float>(1024.0);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = out_ptr1[static_cast<long>(x0)];
                            auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                            auto tmp17 = in_ptr3[static_cast<long>(x2 + (1024L*x0))];
                            auto tmp18 = out_ptr2[static_cast<long>(x0)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = decltype(tmp16)(tmp16 - tmp19);
                            auto tmp21 = decltype(tmp4)(tmp4 * tmp20);
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp23 = static_cast<float>(0.0);
                        auto tmp24 = tmp2 ? tmp22 : tmp23;
                        out_ptr5[static_cast<long>(x2 + (1024L*x1) + (66560L*x0))] = tmp24;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2129920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(65L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (1024L*x2) + (66560L*x1) + (532480L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-532480L) + x3 + (1024L*x2) + (66560L*x1) + (532480L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-1064960L) + x3 + (1024L*x2) + (66560L*x1) + (532480L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (1024L*x0) + (3072L*x2) + (199680L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2129920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(65L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (1024L*x2) + (66560L*x1) + (532480L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-532480L) + x3 + (1024L*x2) + (66560L*x1) + (532480L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-1064960L) + x3 + (1024L*x2) + (66560L*x1) + (532480L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (1024L*x0) + (3072L*x2) + (199680L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2129920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(65L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (1024L*x2) + (66560L*x1) + (532480L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-532480L) + x3 + (1024L*x2) + (66560L*x1) + (532480L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-1064960L) + x3 + (1024L*x2) + (66560L*x1) + (532480L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (1024L*x0) + (3072L*x2) + (199680L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2129920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(65L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (1024L*x2) + (66560L*x1) + (532480L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-532480L) + x3 + (1024L*x2) + (66560L*x1) + (532480L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-1064960L) + x3 + (1024L*x2) + (66560L*x1) + (532480L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (1024L*x0) + (3072L*x2) + (199680L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(520L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp13 = in_ptr4[static_cast<long>(x0)];
                    auto tmp18 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1024.0);
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
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (66560L*x1)));
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


cpp_fused_add_slice_backward_21 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(257L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(256L))) + (131072L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = to_float_mask(tmp2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp6);
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp7);
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (512L*x0)), to_float_mask(tmp10));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                        auto tmp14 = to_float_mask(tmp10);
                        auto tmp15 = decltype(tmp13)::blendv(tmp8, tmp13, tmp14);
                        auto tmp16 = tmp9 + tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (131584L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(512.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_25 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1052672L) + x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2105344L) + x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (512L*x0) + (1536L*x2) + (394752L*x1)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(512.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(512.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_30 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1052672L) + x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2105344L) + x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (512L*x0) + (1536L*x2) + (394752L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(512.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(512.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1052672L) + x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2105344L) + x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (512L*x0) + (1536L*x2) + (394752L*x1)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(512.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(512.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_40 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1052672L) + x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2105344L) + x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (512L*x0) + (1536L*x2) + (394752L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(512.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(512.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_45 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1052672L) + x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2105344L) + x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (512L*x0) + (1536L*x2) + (394752L*x1)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(512.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(512.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1052672L) + x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2105344L) + x3 + (512L*x2) + (131584L*x1) + (1052672L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (512L*x0) + (1536L*x2) + (394752L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2056L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp13 = in_ptr4[static_cast<long>(x0)];
                    auto tmp18 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(512.0);
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
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (131584L*x1)));
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


cpp_fused_add_slice_backward_52 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(962L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(961L))) + (246016L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = to_float_mask(tmp2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp6);
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp7);
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp10));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                        auto tmp14 = to_float_mask(tmp10);
                        auto tmp15 = decltype(tmp13)::blendv(tmp8, tmp13, tmp14);
                        auto tmp16 = tmp9 + tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (246272L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7880704L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_55 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_56 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(962L); x2+=static_cast<long>(1L))
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
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (246272L*x1) + (1970176L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1970176L) + x3 + (256L*x2) + (246272L*x1) + (1970176L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-3940352L) + x3 + (256L*x2) + (246272L*x1) + (1970176L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (256L*x0) + (768L*x2) + (738816L*x1)));
                        }
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7880704L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_60 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_61 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(962L); x2+=static_cast<long>(1L))
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
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (246272L*x1) + (1970176L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1970176L) + x3 + (256L*x2) + (246272L*x1) + (1970176L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-3940352L) + x3 + (256L*x2) + (246272L*x1) + (1970176L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (256L*x0) + (768L*x2) + (738816L*x1)));
                        }
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7880704L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(962L); x2+=static_cast<long>(1L))
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
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (246272L*x1) + (1970176L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1970176L) + x3 + (256L*x2) + (246272L*x1) + (1970176L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-3940352L) + x3 + (256L*x2) + (246272L*x1) + (1970176L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (256L*x0) + (768L*x2) + (738816L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7696L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (246272L*x1)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(256L + x0 + (256L*x1) + (256L*x1_inner) + (246272L*x2))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x1 + (961L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(960L); x1<static_cast<long>(961L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(256L + x0 + (256L*x1) + (246272L*x2))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr6[static_cast<long>(x1 + (961L*x0))] = tmp_acc0;
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
    primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_121, primals_127, primals_133, primals_139, primals_145, primals_151, primals_157, primals_163, primals_169, primals_173, cat, getitem_1, rsqrt, view_1, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, getitem_11, getitem_12, view_5, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_18, getitem_19, getitem_20, getitem_22, getitem_23, getitem_24, getitem_27, getitem_28, view_15, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_34, getitem_35, getitem_36, getitem_38, getitem_39, getitem_40, getitem_43, getitem_44, view_25, mul_16, view_27, addmm_10, view_29, view_31, view_32, cat_1, getitem_49, rsqrt_6, view_35, getitem_50, getitem_51, getitem_52, getitem_54, getitem_55, getitem_56, getitem_59, getitem_60, view_39, mul_23, view_41, addmm_14, view_43, mul_28, view_45, getitem_66, getitem_67, getitem_68, getitem_70, getitem_71, getitem_72, getitem_75, getitem_76, view_49, mul_30, view_51, addmm_18, view_53, mul_35, view_55, getitem_82, getitem_83, getitem_84, getitem_86, getitem_87, getitem_88, getitem_91, getitem_92, view_59, mul_37, view_61, addmm_22, view_63, mul_42, view_65, getitem_98, getitem_99, getitem_100, getitem_102, getitem_103, getitem_104, getitem_107, getitem_108, view_69, mul_44, view_71, addmm_26, view_73, mul_49, view_75, getitem_114, getitem_115, getitem_116, getitem_118, getitem_119, getitem_120, getitem_123, getitem_124, view_79, mul_51, view_81, addmm_30, view_83, mul_56, view_85, getitem_130, getitem_131, getitem_132, getitem_134, getitem_135, getitem_136, getitem_139, getitem_140, view_89, mul_58, view_91, addmm_34, view_93, view_95, view_96, cat_2, getitem_145, rsqrt_18, view_99, getitem_146, getitem_147, getitem_148, getitem_150, getitem_151, getitem_152, getitem_155, getitem_156, view_103, mul_65, view_105, addmm_38, view_107, mul_70, view_109, getitem_162, getitem_163, getitem_164, getitem_166, getitem_167, getitem_168, getitem_171, getitem_172, view_113, mul_72, view_115, addmm_42, view_117, mul_77, view_119, getitem_178, getitem_179, getitem_180, getitem_182, getitem_183, getitem_184, getitem_187, getitem_188, view_123, mul_79, view_125, addmm_46, view_127, mul_84, view_129, getitem_194, getitem_195, getitem_196, getitem_198, getitem_199, getitem_200, getitem_203, getitem_204, view_133, mul_86, view_135, addmm_50, view_137, mul_91, clone_41, permute_87, div, permute_91, permute_95, div_1, permute_99, alias_13, permute_105, div_2, permute_109, permute_113, div_3, permute_117, alias_14, permute_123, div_4, permute_127, permute_131, div_5, permute_135, alias_15, permute_141, div_6, permute_145, permute_149, div_7, permute_153, alias_16, permute_159, permute_166, permute_169, permute_173, div_9, permute_177, alias_17, permute_183, div_10, permute_187, permute_191, div_11, permute_195, alias_18, permute_201, div_12, permute_205, permute_209, div_13, permute_213, alias_19, permute_219, div_14, permute_223, permute_227, div_15, permute_231, alias_20, permute_237, div_16, permute_241, permute_245, div_17, permute_249, alias_21, permute_255, div_18, permute_259, permute_263, div_19, permute_267, alias_22, permute_273, permute_280, permute_283, permute_287, div_21, permute_291, alias_23, permute_297, div_22, permute_301, permute_305, div_23, permute_309, alias_24, permute_315, div_24, permute_319, permute_323, div_25, permute_327, alias_25, permute_333, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (256, 3, 14, 14), (588, 1, 42, 3))
    assert_size_stride(primals_5, (256, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_41, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_111, (512, ), (1, ))
    assert_size_stride(primals_117, (1024, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_121, (1024, ), (1, ))
    assert_size_stride(primals_127, (1024, ), (1, ))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_139, (1024, ), (1, ))
    assert_size_stride(primals_145, (1024, ), (1, ))
    assert_size_stride(primals_151, (1024, ), (1, ))
    assert_size_stride(primals_157, (1024, ), (1, ))
    assert_size_stride(primals_163, (1024, ), (1, ))
    assert_size_stride(primals_169, (1024, ), (1, ))
    assert_size_stride(primals_173, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(cat, (8, 962, 256), (246272, 256, 1))
    assert_size_stride(getitem_1, (8, 962, 1), (962, 1, 1))
    assert_size_stride(rsqrt, (8, 962, 1), (962, 1, 1))
    assert_size_stride(view_1, (7696, 256), (256, 1))
    assert_size_stride(getitem_2, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_3, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_4, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_6, (8, 4, 962), (3848, 1, 4))
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(getitem_8, (), ())
    assert_size_stride(getitem_11, (), ())
    assert_size_stride(getitem_12, (), ())
    assert_size_stride(view_5, (7696, 256), (256, 1))
    assert_size_stride(mul_2, (8, 962, 256), (246272, 256, 1))
    assert_size_stride(view_7, (7696, 256), (256, 1))
    assert_size_stride(addmm_2, (7696, 1024), (1024, 1))
    assert_size_stride(view_9, (7696, 1024), (1024, 1))
    assert_size_stride(mul_7, (8, 962, 256), (246272, 256, 1))
    assert_size_stride(view_11, (7696, 256), (256, 1))
    assert_size_stride(getitem_18, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_19, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_20, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_22, (8, 4, 962), (3848, 1, 4))
    assert_size_stride(getitem_23, (), ())
    assert_size_stride(getitem_24, (), ())
    assert_size_stride(getitem_27, (), ())
    assert_size_stride(getitem_28, (), ())
    assert_size_stride(view_15, (7696, 256), (256, 1))
    assert_size_stride(mul_9, (8, 962, 256), (246272, 256, 1))
    assert_size_stride(view_17, (7696, 256), (256, 1))
    assert_size_stride(addmm_6, (7696, 1024), (1024, 1))
    assert_size_stride(view_19, (7696, 1024), (1024, 1))
    assert_size_stride(mul_14, (8, 962, 256), (246272, 256, 1))
    assert_size_stride(view_21, (7696, 256), (256, 1))
    assert_size_stride(getitem_34, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_35, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_36, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_38, (8, 4, 962), (3848, 1, 4))
    assert_size_stride(getitem_39, (), ())
    assert_size_stride(getitem_40, (), ())
    assert_size_stride(getitem_43, (), ())
    assert_size_stride(getitem_44, (), ())
    assert_size_stride(view_25, (7696, 256), (256, 1))
    assert_size_stride(mul_16, (8, 962, 256), (246272, 256, 1))
    assert_size_stride(view_27, (7696, 256), (256, 1))
    assert_size_stride(addmm_10, (7696, 1024), (1024, 1))
    assert_size_stride(view_29, (7696, 1024), (1024, 1))
    assert_size_stride(view_31, (8, 256, 31, 31), (246272, 1, 7936, 256))
    assert_size_stride(view_32, (8, 256), (246272, 1))
    assert_size_stride(cat_1, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(getitem_49, (8, 257, 1), (257, 1, 1))
    assert_size_stride(rsqrt_6, (8, 257, 1), (257, 1, 1))
    assert_size_stride(view_35, (2056, 512), (512, 1))
    assert_size_stride(getitem_50, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_51, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_52, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_54, (8, 8, 257), (2056, 1, 8))
    assert_size_stride(getitem_55, (), ())
    assert_size_stride(getitem_56, (), ())
    assert_size_stride(getitem_59, (), ())
    assert_size_stride(getitem_60, (), ())
    assert_size_stride(view_39, (2056, 512), (512, 1))
    assert_size_stride(mul_23, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_41, (2056, 512), (512, 1))
    assert_size_stride(addmm_14, (2056, 2048), (2048, 1))
    assert_size_stride(view_43, (2056, 2048), (2048, 1))
    assert_size_stride(mul_28, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_45, (2056, 512), (512, 1))
    assert_size_stride(getitem_66, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_67, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_68, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_70, (8, 8, 257), (2056, 1, 8))
    assert_size_stride(getitem_71, (), ())
    assert_size_stride(getitem_72, (), ())
    assert_size_stride(getitem_75, (), ())
    assert_size_stride(getitem_76, (), ())
    assert_size_stride(view_49, (2056, 512), (512, 1))
    assert_size_stride(mul_30, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_51, (2056, 512), (512, 1))
    assert_size_stride(addmm_18, (2056, 2048), (2048, 1))
    assert_size_stride(view_53, (2056, 2048), (2048, 1))
    assert_size_stride(mul_35, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_55, (2056, 512), (512, 1))
    assert_size_stride(getitem_82, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_83, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_84, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_86, (8, 8, 257), (2056, 1, 8))
    assert_size_stride(getitem_87, (), ())
    assert_size_stride(getitem_88, (), ())
    assert_size_stride(getitem_91, (), ())
    assert_size_stride(getitem_92, (), ())
    assert_size_stride(view_59, (2056, 512), (512, 1))
    assert_size_stride(mul_37, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_61, (2056, 512), (512, 1))
    assert_size_stride(addmm_22, (2056, 2048), (2048, 1))
    assert_size_stride(view_63, (2056, 2048), (2048, 1))
    assert_size_stride(mul_42, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_65, (2056, 512), (512, 1))
    assert_size_stride(getitem_98, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_99, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_100, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_102, (8, 8, 257), (2056, 1, 8))
    assert_size_stride(getitem_103, (), ())
    assert_size_stride(getitem_104, (), ())
    assert_size_stride(getitem_107, (), ())
    assert_size_stride(getitem_108, (), ())
    assert_size_stride(view_69, (2056, 512), (512, 1))
    assert_size_stride(mul_44, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_71, (2056, 512), (512, 1))
    assert_size_stride(addmm_26, (2056, 2048), (2048, 1))
    assert_size_stride(view_73, (2056, 2048), (2048, 1))
    assert_size_stride(mul_49, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_75, (2056, 512), (512, 1))
    assert_size_stride(getitem_114, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_115, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_116, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_118, (8, 8, 257), (2056, 1, 8))
    assert_size_stride(getitem_119, (), ())
    assert_size_stride(getitem_120, (), ())
    assert_size_stride(getitem_123, (), ())
    assert_size_stride(getitem_124, (), ())
    assert_size_stride(view_79, (2056, 512), (512, 1))
    assert_size_stride(mul_51, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_81, (2056, 512), (512, 1))
    assert_size_stride(addmm_30, (2056, 2048), (2048, 1))
    assert_size_stride(view_83, (2056, 2048), (2048, 1))
    assert_size_stride(mul_56, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_85, (2056, 512), (512, 1))
    assert_size_stride(getitem_130, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_131, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_132, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_134, (8, 8, 257), (2056, 1, 8))
    assert_size_stride(getitem_135, (), ())
    assert_size_stride(getitem_136, (), ())
    assert_size_stride(getitem_139, (), ())
    assert_size_stride(getitem_140, (), ())
    assert_size_stride(view_89, (2056, 512), (512, 1))
    assert_size_stride(mul_58, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_91, (2056, 512), (512, 1))
    assert_size_stride(addmm_34, (2056, 2048), (2048, 1))
    assert_size_stride(view_93, (2056, 2048), (2048, 1))
    assert_size_stride(view_95, (8, 512, 16, 16), (131584, 1, 8192, 512))
    assert_size_stride(view_96, (8, 512), (131584, 1))
    assert_size_stride(cat_2, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(getitem_145, (8, 65, 1), (65, 1, 1))
    assert_size_stride(rsqrt_18, (8, 65, 1), (65, 1, 1))
    assert_size_stride(view_99, (520, 1024), (1024, 1))
    assert_size_stride(getitem_146, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_147, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_148, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_150, (8, 16, 65), (1040, 1, 16))
    assert_size_stride(getitem_151, (), ())
    assert_size_stride(getitem_152, (), ())
    assert_size_stride(getitem_155, (), ())
    assert_size_stride(getitem_156, (), ())
    assert_size_stride(view_103, (520, 1024), (1024, 1))
    assert_size_stride(mul_65, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_105, (520, 1024), (1024, 1))
    assert_size_stride(addmm_38, (520, 4096), (4096, 1))
    assert_size_stride(view_107, (520, 4096), (4096, 1))
    assert_size_stride(mul_70, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_109, (520, 1024), (1024, 1))
    assert_size_stride(getitem_162, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_163, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_164, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_166, (8, 16, 65), (1040, 1, 16))
    assert_size_stride(getitem_167, (), ())
    assert_size_stride(getitem_168, (), ())
    assert_size_stride(getitem_171, (), ())
    assert_size_stride(getitem_172, (), ())
    assert_size_stride(view_113, (520, 1024), (1024, 1))
    assert_size_stride(mul_72, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_115, (520, 1024), (1024, 1))
    assert_size_stride(addmm_42, (520, 4096), (4096, 1))
    assert_size_stride(view_117, (520, 4096), (4096, 1))
    assert_size_stride(mul_77, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_119, (520, 1024), (1024, 1))
    assert_size_stride(getitem_178, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_179, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_180, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_182, (8, 16, 65), (1040, 1, 16))
    assert_size_stride(getitem_183, (), ())
    assert_size_stride(getitem_184, (), ())
    assert_size_stride(getitem_187, (), ())
    assert_size_stride(getitem_188, (), ())
    assert_size_stride(view_123, (520, 1024), (1024, 1))
    assert_size_stride(mul_79, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_125, (520, 1024), (1024, 1))
    assert_size_stride(addmm_46, (520, 4096), (4096, 1))
    assert_size_stride(view_127, (520, 4096), (4096, 1))
    assert_size_stride(mul_84, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_129, (520, 1024), (1024, 1))
    assert_size_stride(getitem_194, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_195, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_196, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_198, (8, 16, 65), (1040, 1, 16))
    assert_size_stride(getitem_199, (), ())
    assert_size_stride(getitem_200, (), ())
    assert_size_stride(getitem_203, (), ())
    assert_size_stride(getitem_204, (), ())
    assert_size_stride(view_133, (520, 1024), (1024, 1))
    assert_size_stride(mul_86, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_135, (520, 1024), (1024, 1))
    assert_size_stride(addmm_50, (520, 4096), (4096, 1))
    assert_size_stride(view_137, (520, 4096), (4096, 1))
    assert_size_stride(mul_91, (8, 1, 1024), (1024, 1024, 1))
    assert_size_stride(clone_41, (8, 1024), (1024, 1))
    assert_size_stride(permute_87, (1000, 1024), (1024, 1))
    assert_size_stride(div, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_91, (1024, 4096), (4096, 1))
    assert_size_stride(permute_95, (4096, 1024), (1024, 1))
    assert_size_stride(div_1, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_99, (1024, 1024), (1024, 1))
    assert_size_stride(alias_13, (8, 16, 65, 64), (66560, 1, 1024, 16))
    assert_size_stride(permute_105, (3072, 1024), (1024, 1))
    assert_size_stride(div_2, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_109, (1024, 4096), (4096, 1))
    assert_size_stride(permute_113, (4096, 1024), (1024, 1))
    assert_size_stride(div_3, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_117, (1024, 1024), (1024, 1))
    assert_size_stride(alias_14, (8, 16, 65, 64), (66560, 1, 1024, 16))
    assert_size_stride(permute_123, (3072, 1024), (1024, 1))
    assert_size_stride(div_4, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_127, (1024, 4096), (4096, 1))
    assert_size_stride(permute_131, (4096, 1024), (1024, 1))
    assert_size_stride(div_5, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_135, (1024, 1024), (1024, 1))
    assert_size_stride(alias_15, (8, 16, 65, 64), (66560, 1, 1024, 16))
    assert_size_stride(permute_141, (3072, 1024), (1024, 1))
    assert_size_stride(div_6, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_145, (1024, 4096), (4096, 1))
    assert_size_stride(permute_149, (4096, 1024), (1024, 1))
    assert_size_stride(div_7, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_153, (1024, 1024), (1024, 1))
    assert_size_stride(alias_16, (8, 16, 65, 64), (66560, 1, 1024, 16))
    assert_size_stride(permute_159, (3072, 1024), (1024, 1))
    assert_size_stride(permute_166, (1024, 512), (512, 1))
    assert_size_stride(permute_169, (512, 2048), (2048, 1))
    assert_size_stride(permute_173, (2048, 512), (512, 1))
    assert_size_stride(div_9, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_177, (512, 512), (512, 1))
    assert_size_stride(alias_17, (8, 8, 257, 64), (131584, 1, 512, 8))
    assert_size_stride(permute_183, (1536, 512), (512, 1))
    assert_size_stride(div_10, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_187, (512, 2048), (2048, 1))
    assert_size_stride(permute_191, (2048, 512), (512, 1))
    assert_size_stride(div_11, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_195, (512, 512), (512, 1))
    assert_size_stride(alias_18, (8, 8, 257, 64), (131584, 1, 512, 8))
    assert_size_stride(permute_201, (1536, 512), (512, 1))
    assert_size_stride(div_12, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_205, (512, 2048), (2048, 1))
    assert_size_stride(permute_209, (2048, 512), (512, 1))
    assert_size_stride(div_13, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_213, (512, 512), (512, 1))
    assert_size_stride(alias_19, (8, 8, 257, 64), (131584, 1, 512, 8))
    assert_size_stride(permute_219, (1536, 512), (512, 1))
    assert_size_stride(div_14, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_223, (512, 2048), (2048, 1))
    assert_size_stride(permute_227, (2048, 512), (512, 1))
    assert_size_stride(div_15, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_231, (512, 512), (512, 1))
    assert_size_stride(alias_20, (8, 8, 257, 64), (131584, 1, 512, 8))
    assert_size_stride(permute_237, (1536, 512), (512, 1))
    assert_size_stride(div_16, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_241, (512, 2048), (2048, 1))
    assert_size_stride(permute_245, (2048, 512), (512, 1))
    assert_size_stride(div_17, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_249, (512, 512), (512, 1))
    assert_size_stride(alias_21, (8, 8, 257, 64), (131584, 1, 512, 8))
    assert_size_stride(permute_255, (1536, 512), (512, 1))
    assert_size_stride(div_18, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_259, (512, 2048), (2048, 1))
    assert_size_stride(permute_263, (2048, 512), (512, 1))
    assert_size_stride(div_19, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_267, (512, 512), (512, 1))
    assert_size_stride(alias_22, (8, 8, 257, 64), (131584, 1, 512, 8))
    assert_size_stride(permute_273, (1536, 512), (512, 1))
    assert_size_stride(permute_280, (512, 256), (256, 1))
    assert_size_stride(permute_283, (256, 1024), (1024, 1))
    assert_size_stride(permute_287, (1024, 256), (256, 1))
    assert_size_stride(div_21, (8, 962, 1), (962, 1, 1))
    assert_size_stride(permute_291, (256, 256), (256, 1))
    assert_size_stride(alias_23, (8, 4, 962, 64), (246272, 1, 256, 4))
    assert_size_stride(permute_297, (768, 256), (256, 1))
    assert_size_stride(div_22, (8, 962, 1), (962, 1, 1))
    assert_size_stride(permute_301, (256, 1024), (1024, 1))
    assert_size_stride(permute_305, (1024, 256), (256, 1))
    assert_size_stride(div_23, (8, 962, 1), (962, 1, 1))
    assert_size_stride(permute_309, (256, 256), (256, 1))
    assert_size_stride(alias_24, (8, 4, 962, 64), (246272, 1, 256, 4))
    assert_size_stride(permute_315, (768, 256), (256, 1))
    assert_size_stride(div_24, (8, 962, 1), (962, 1, 1))
    assert_size_stride(permute_319, (256, 1024), (1024, 1))
    assert_size_stride(permute_323, (1024, 256), (256, 1))
    assert_size_stride(div_25, (8, 962, 1), (962, 1, 1))
    assert_size_stride(permute_327, (256, 256), (256, 1))
    assert_size_stride(alias_25, (8, 4, 962, 64), (246272, 1, 256, 4))
    assert_size_stride(permute_333, (768, 256), (256, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_87, out=buf0)
    del permute_87
    buf1 = empty((1000, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_41, out=buf1)
    del clone_41
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf5 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf6 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf7 = empty((8, 65, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_select_backward_slice_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(mul_91.data_ptr()), c_void_p(div.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del buf0
    del buf3
    del buf4
    del div
    del mul_91
    del primals_169
    del tangents_1
    buf8 = empty((520, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf7, (520, 1024), (1024, 1), 0), permute_91, out=buf8)
    del permute_91
    buf9 = empty((1024, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf7, (1024, 520), (1, 1024), 0), view_137, out=buf9)
    del view_137
    buf10 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf11 = reinterpret_tensor(buf8, (8, 65, 4096), (266240, 4096, 1), 0); del buf8  # reuse
    cpp_fused_gelu_gelu_backward_sum_1(c_void_p(buf11.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(addmm_50.data_ptr()), c_void_p(buf10.data_ptr()))
    del addmm_50
    buf12 = empty((520, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (520, 4096), (4096, 1), 0), permute_95, out=buf12)
    del permute_95
    buf13 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (4096, 520), (1, 4096), 0), view_135, out=buf13)
    del view_135
    buf14 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf17 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf18 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf19 = reinterpret_tensor(buf12, (8, 65, 1024), (66560, 1024, 1), 0); del buf12  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_2(c_void_p(buf19.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(mul_86.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    del div_1
    del mul_86
    del primals_163
    buf20 = reinterpret_tensor(buf7, (520, 1024), (1024, 1), 0); del buf7  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf19, (520, 1024), (1024, 1), 0), permute_99, out=buf20)
    del permute_99
    buf21 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf19, (1024, 520), (1, 1024), 0), view_133, out=buf21)
    del view_133
    buf22 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_3(c_void_p(buf19.data_ptr()), c_void_p(buf22.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf23 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf20, (8, 16, 65, 64), (66560, 64, 1024, 1), 0), getitem_194, getitem_195, getitem_196, alias_13, getitem_198, getitem_199, getitem_200, 0, 0, 0.0, False, getitem_203, getitem_204)
    del alias_13
    del buf20
    del getitem_194
    del getitem_195
    del getitem_196
    del getitem_198
    del getitem_199
    del getitem_200
    del getitem_203
    del getitem_204
    buf24 = buf23[0]
    buf25 = buf23[1]
    buf26 = buf23[2]
    del buf23
    buf27 = empty((8, 65, 3, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_4(c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    del buf24
    del buf25
    buf28 = reinterpret_tensor(buf26, (520, 1024), (1024, 1), 0); del buf26  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf27, (520, 3072), (3072, 1), 0), permute_105, out=buf28)
    del permute_105
    buf29 = empty((3072, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf27, (3072, 520), (1, 3072), 0), view_129, out=buf29)
    del view_129
    buf30 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf31 = buf16; del buf16  # reuse
    buf32 = buf15; del buf15  # reuse
    buf33 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf34 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf35 = buf19; del buf19  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_5(c_void_p(buf35.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(mul_84.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del div_2
    del mul_84
    del primals_157
    buf36 = reinterpret_tensor(buf11, (520, 4096), (4096, 1), 0); del buf11  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (520, 1024), (1024, 1), 0), permute_109, out=buf36)
    del permute_109
    buf37 = empty((1024, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (1024, 520), (1, 1024), 0), view_127, out=buf37)
    del view_127
    buf38 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf39 = reinterpret_tensor(buf36, (8, 65, 4096), (266240, 4096, 1), 0); del buf36  # reuse
    cpp_fused_gelu_gelu_backward_sum_6(c_void_p(buf39.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(buf38.data_ptr()))
    del addmm_46
    buf40 = buf28; del buf28  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (520, 4096), (4096, 1), 0), permute_113, out=buf40)
    del permute_113
    buf41 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (4096, 520), (1, 4096), 0), view_125, out=buf41)
    del view_125
    buf42 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf43 = buf32; del buf32  # reuse
    buf44 = buf31; del buf31  # reuse
    buf45 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf46 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf47 = buf35; del buf35  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_7(c_void_p(buf47.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(mul_79.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    del div_3
    del mul_79
    del primals_151
    buf48 = buf40; del buf40  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (520, 1024), (1024, 1), 0), permute_117, out=buf48)
    del permute_117
    buf49 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (1024, 520), (1, 1024), 0), view_123, out=buf49)
    del view_123
    buf50 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_8(c_void_p(buf47.data_ptr()), c_void_p(buf50.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf51 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf48, (8, 16, 65, 64), (66560, 64, 1024, 1), 0), getitem_178, getitem_179, getitem_180, alias_14, getitem_182, getitem_183, getitem_184, 0, 0, 0.0, False, getitem_187, getitem_188)
    del alias_14
    del buf48
    del getitem_178
    del getitem_179
    del getitem_180
    del getitem_182
    del getitem_183
    del getitem_184
    del getitem_187
    del getitem_188
    buf52 = buf51[0]
    buf53 = buf51[1]
    buf54 = buf51[2]
    del buf51
    buf55 = buf27; del buf27  # reuse
    cpp_fused_clone_9(c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del buf52
    del buf53
    buf56 = reinterpret_tensor(buf54, (520, 1024), (1024, 1), 0); del buf54  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf55, (520, 3072), (3072, 1), 0), permute_123, out=buf56)
    del permute_123
    buf57 = empty((3072, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf55, (3072, 520), (1, 3072), 0), view_119, out=buf57)
    del view_119
    buf58 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf59 = buf44; del buf44  # reuse
    buf60 = buf43; del buf43  # reuse
    buf61 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf62 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf63 = buf47; del buf47  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_10(c_void_p(buf63.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(mul_77.data_ptr()), c_void_p(div_4.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del div_4
    del mul_77
    del primals_145
    buf64 = reinterpret_tensor(buf39, (520, 4096), (4096, 1), 0); del buf39  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf63, (520, 1024), (1024, 1), 0), permute_127, out=buf64)
    del permute_127
    buf65 = empty((1024, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf63, (1024, 520), (1, 1024), 0), view_117, out=buf65)
    del view_117
    buf66 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf67 = reinterpret_tensor(buf64, (8, 65, 4096), (266240, 4096, 1), 0); del buf64  # reuse
    cpp_fused_gelu_gelu_backward_sum_11(c_void_p(buf67.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(addmm_42.data_ptr()), c_void_p(buf66.data_ptr()))
    del addmm_42
    buf68 = buf56; del buf56  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (520, 4096), (4096, 1), 0), permute_131, out=buf68)
    del permute_131
    buf69 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (4096, 520), (1, 4096), 0), view_115, out=buf69)
    del view_115
    buf70 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf71 = buf60; del buf60  # reuse
    buf72 = buf59; del buf59  # reuse
    buf73 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf74 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf75 = buf63; del buf63  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_12(c_void_p(buf75.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(mul_72.data_ptr()), c_void_p(div_5.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    del div_5
    del mul_72
    del primals_139
    buf76 = buf68; del buf68  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf75, (520, 1024), (1024, 1), 0), permute_135, out=buf76)
    del permute_135
    buf77 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf75, (1024, 520), (1, 1024), 0), view_113, out=buf77)
    del view_113
    buf78 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_13(c_void_p(buf75.data_ptr()), c_void_p(buf78.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf79 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf76, (8, 16, 65, 64), (66560, 64, 1024, 1), 0), getitem_162, getitem_163, getitem_164, alias_15, getitem_166, getitem_167, getitem_168, 0, 0, 0.0, False, getitem_171, getitem_172)
    del alias_15
    del buf76
    del getitem_162
    del getitem_163
    del getitem_164
    del getitem_166
    del getitem_167
    del getitem_168
    del getitem_171
    del getitem_172
    buf80 = buf79[0]
    buf81 = buf79[1]
    buf82 = buf79[2]
    del buf79
    buf83 = buf55; del buf55  # reuse
    cpp_fused_clone_14(c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    del buf80
    del buf81
    buf84 = reinterpret_tensor(buf82, (520, 1024), (1024, 1), 0); del buf82  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (520, 3072), (3072, 1), 0), permute_141, out=buf84)
    del permute_141
    buf85 = empty((3072, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (3072, 520), (1, 3072), 0), view_109, out=buf85)
    del view_109
    buf86 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf87 = buf72; del buf72  # reuse
    buf88 = buf71; del buf71  # reuse
    buf89 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf90 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf91 = buf75; del buf75  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_15(c_void_p(buf91.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(mul_70.data_ptr()), c_void_p(div_6.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del div_6
    del mul_70
    del primals_133
    buf92 = reinterpret_tensor(buf67, (520, 4096), (4096, 1), 0); del buf67  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (520, 1024), (1024, 1), 0), permute_145, out=buf92)
    del permute_145
    buf93 = empty((1024, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (1024, 520), (1, 1024), 0), view_107, out=buf93)
    del view_107
    buf94 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf95 = reinterpret_tensor(buf92, (8, 65, 4096), (266240, 4096, 1), 0); del buf92  # reuse
    cpp_fused_gelu_gelu_backward_sum_16(c_void_p(buf95.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(addmm_38.data_ptr()), c_void_p(buf94.data_ptr()))
    del addmm_38
    buf96 = buf84; del buf84  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf95, (520, 4096), (4096, 1), 0), permute_149, out=buf96)
    del permute_149
    buf97 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf95, (4096, 520), (1, 4096), 0), view_105, out=buf97)
    del view_105
    buf98 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf99 = buf88; del buf88  # reuse
    buf100 = buf87; del buf87  # reuse
    buf101 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf102 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf103 = buf91; del buf91  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_17(c_void_p(buf103.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(mul_65.data_ptr()), c_void_p(div_7.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    del buf95
    del div_7
    del mul_65
    del primals_127
    buf104 = buf96; del buf96  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (520, 1024), (1024, 1), 0), permute_153, out=buf104)
    del permute_153
    buf105 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (1024, 520), (1, 1024), 0), view_103, out=buf105)
    del view_103
    buf106 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_18(c_void_p(buf103.data_ptr()), c_void_p(buf106.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf107 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf104, (8, 16, 65, 64), (66560, 64, 1024, 1), 0), getitem_146, getitem_147, getitem_148, alias_16, getitem_150, getitem_151, getitem_152, 0, 0, 0.0, False, getitem_155, getitem_156)
    del alias_16
    del buf104
    del getitem_146
    del getitem_147
    del getitem_148
    del getitem_150
    del getitem_151
    del getitem_152
    del getitem_155
    del getitem_156
    buf108 = buf107[0]
    buf109 = buf107[1]
    buf110 = buf107[2]
    del buf107
    buf111 = buf83; del buf83  # reuse
    cpp_fused_clone_19(c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    del buf108
    del buf109
    buf112 = reinterpret_tensor(buf110, (520, 1024), (1024, 1), 0); del buf110  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf111, (520, 3072), (3072, 1), 0), permute_159, out=buf112)
    del permute_159
    buf113 = empty((3072, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf111, (3072, 520), (1, 3072), 0), view_99, out=buf113)
    del view_99
    buf114 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf115 = buf99; del buf99  # reuse
    buf116 = buf100; del buf100  # reuse
    buf117 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf118 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf119 = buf103; del buf103  # reuse
    buf120 = empty((1, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_20(c_void_p(buf119.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(cat_2.data_ptr()), c_void_p(getitem_145.data_ptr()), c_void_p(rsqrt_18.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()))
    del buf111
    del buf112
    del buf115
    del buf116
    del cat_2
    del getitem_145
    del primals_121
    del rsqrt_18
    buf121 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (1024, 8), (1, 66560), 0), view_96, out=buf121)
    del view_96
    buf122 = empty((8, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (8, 1024), (66560, 1), 0), permute_166, out=buf122)
    del permute_166
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf123 = aten.convolution_backward(reinterpret_tensor(buf119, (8, 1024, 8, 8), (66560, 1, 8192, 1024), 1024), view_95, primals_117, [1024], [2, 2], [1, 1], [1, 1], False, [0, 0], 512, [True, True, True])
    del buf119
    del primals_117
    del view_95
    buf124 = buf123[0]
    buf125 = buf123[1]
    buf126 = buf123[2]
    del buf123
    buf127 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_slice_backward_21(c_void_p(buf124.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf127.data_ptr()))
    del buf122
    buf128 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf127, (2056, 512), (512, 1), 0), permute_169, out=buf128)
    del permute_169
    buf129 = reinterpret_tensor(buf124, (512, 2048), (2048, 1), 0); del buf124  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf127, (512, 2056), (1, 512), 0), view_93, out=buf129)
    del view_93
    buf130 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf131 = reinterpret_tensor(buf128, (8, 257, 2048), (526336, 2048, 1), 0); del buf128  # reuse
    cpp_fused_gelu_gelu_backward_sum_22(c_void_p(buf131.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf130.data_ptr()))
    del addmm_34
    buf132 = empty((2056, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf131, (2056, 2048), (2048, 1), 0), permute_173, out=buf132)
    del permute_173
    buf133 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf131, (2048, 2056), (1, 2048), 0), view_91, out=buf133)
    del view_91
    buf134 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf135 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf136 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf137 = empty((512, ), device='cpu', dtype=torch.float32)
    buf138 = empty((512, ), device='cpu', dtype=torch.float32)
    buf139 = buf127; del buf127  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_23(c_void_p(buf139.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(mul_58.data_ptr()), c_void_p(div_9.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()))
    del div_9
    del mul_58
    del primals_111
    buf140 = buf132; del buf132  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (2056, 512), (512, 1), 0), permute_177, out=buf140)
    del permute_177
    buf141 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (512, 2056), (1, 512), 0), view_89, out=buf141)
    del view_89
    buf142 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_24(c_void_p(buf139.data_ptr()), c_void_p(buf142.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf143 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf140, (8, 8, 257, 64), (131584, 64, 512, 1), 0), getitem_130, getitem_131, getitem_132, alias_17, getitem_134, getitem_135, getitem_136, 0, 0, 0.0, False, getitem_139, getitem_140)
    del alias_17
    del buf140
    del getitem_130
    del getitem_131
    del getitem_132
    del getitem_134
    del getitem_135
    del getitem_136
    del getitem_139
    del getitem_140
    buf144 = buf143[0]
    buf145 = buf143[1]
    buf146 = buf143[2]
    del buf143
    buf147 = empty((8, 257, 3, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_25(c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    del buf144
    del buf145
    buf148 = reinterpret_tensor(buf146, (2056, 512), (512, 1), 0); del buf146  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf147, (2056, 1536), (1536, 1), 0), permute_183, out=buf148)
    del permute_183
    buf149 = empty((1536, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf147, (1536, 2056), (1, 1536), 0), view_85, out=buf149)
    del view_85
    buf150 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf151 = buf136; del buf136  # reuse
    buf152 = buf135; del buf135  # reuse
    buf153 = empty((512, ), device='cpu', dtype=torch.float32)
    buf154 = empty((512, ), device='cpu', dtype=torch.float32)
    buf155 = buf139; del buf139  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_26(c_void_p(buf155.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(mul_56.data_ptr()), c_void_p(div_10.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    del div_10
    del mul_56
    del primals_105
    buf156 = reinterpret_tensor(buf131, (2056, 2048), (2048, 1), 0); del buf131  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf155, (2056, 512), (512, 1), 0), permute_187, out=buf156)
    del permute_187
    buf157 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf155, (512, 2056), (1, 512), 0), view_83, out=buf157)
    del view_83
    buf158 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf159 = reinterpret_tensor(buf156, (8, 257, 2048), (526336, 2048, 1), 0); del buf156  # reuse
    cpp_fused_gelu_gelu_backward_sum_27(c_void_p(buf159.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(addmm_30.data_ptr()), c_void_p(buf158.data_ptr()))
    del addmm_30
    buf160 = buf148; del buf148  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (2056, 2048), (2048, 1), 0), permute_191, out=buf160)
    del permute_191
    buf161 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (2048, 2056), (1, 2048), 0), view_81, out=buf161)
    del view_81
    buf162 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf163 = buf152; del buf152  # reuse
    buf164 = buf151; del buf151  # reuse
    buf165 = empty((512, ), device='cpu', dtype=torch.float32)
    buf166 = empty((512, ), device='cpu', dtype=torch.float32)
    buf167 = buf155; del buf155  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_28(c_void_p(buf167.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(mul_51.data_ptr()), c_void_p(div_11.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    del div_11
    del mul_51
    del primals_99
    buf168 = buf160; del buf160  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (2056, 512), (512, 1), 0), permute_195, out=buf168)
    del permute_195
    buf169 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (512, 2056), (1, 512), 0), view_79, out=buf169)
    del view_79
    buf170 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_29(c_void_p(buf167.data_ptr()), c_void_p(buf170.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf171 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf168, (8, 8, 257, 64), (131584, 64, 512, 1), 0), getitem_114, getitem_115, getitem_116, alias_18, getitem_118, getitem_119, getitem_120, 0, 0, 0.0, False, getitem_123, getitem_124)
    del alias_18
    del buf168
    del getitem_114
    del getitem_115
    del getitem_116
    del getitem_118
    del getitem_119
    del getitem_120
    del getitem_123
    del getitem_124
    buf172 = buf171[0]
    buf173 = buf171[1]
    buf174 = buf171[2]
    del buf171
    buf175 = buf147; del buf147  # reuse
    cpp_fused_clone_30(c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    del buf172
    del buf173
    buf176 = reinterpret_tensor(buf174, (2056, 512), (512, 1), 0); del buf174  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf175, (2056, 1536), (1536, 1), 0), permute_201, out=buf176)
    del permute_201
    buf177 = empty((1536, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf175, (1536, 2056), (1, 1536), 0), view_75, out=buf177)
    del view_75
    buf178 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf179 = buf164; del buf164  # reuse
    buf180 = buf163; del buf163  # reuse
    buf181 = empty((512, ), device='cpu', dtype=torch.float32)
    buf182 = empty((512, ), device='cpu', dtype=torch.float32)
    buf183 = buf167; del buf167  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_31(c_void_p(buf183.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(mul_49.data_ptr()), c_void_p(div_12.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()))
    del div_12
    del mul_49
    del primals_93
    buf184 = reinterpret_tensor(buf159, (2056, 2048), (2048, 1), 0); del buf159  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf183, (2056, 512), (512, 1), 0), permute_205, out=buf184)
    del permute_205
    buf185 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf183, (512, 2056), (1, 512), 0), view_73, out=buf185)
    del view_73
    buf186 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf187 = reinterpret_tensor(buf184, (8, 257, 2048), (526336, 2048, 1), 0); del buf184  # reuse
    cpp_fused_gelu_gelu_backward_sum_32(c_void_p(buf187.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(addmm_26.data_ptr()), c_void_p(buf186.data_ptr()))
    del addmm_26
    buf188 = buf176; del buf176  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf187, (2056, 2048), (2048, 1), 0), permute_209, out=buf188)
    del permute_209
    buf189 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf187, (2048, 2056), (1, 2048), 0), view_71, out=buf189)
    del view_71
    buf190 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf191 = buf180; del buf180  # reuse
    buf192 = buf179; del buf179  # reuse
    buf193 = empty((512, ), device='cpu', dtype=torch.float32)
    buf194 = empty((512, ), device='cpu', dtype=torch.float32)
    buf195 = buf183; del buf183  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_33(c_void_p(buf195.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(mul_44.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()))
    del div_13
    del mul_44
    del primals_87
    buf196 = buf188; del buf188  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (2056, 512), (512, 1), 0), permute_213, out=buf196)
    del permute_213
    buf197 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (512, 2056), (1, 512), 0), view_69, out=buf197)
    del view_69
    buf198 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_34(c_void_p(buf195.data_ptr()), c_void_p(buf198.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf199 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf196, (8, 8, 257, 64), (131584, 64, 512, 1), 0), getitem_98, getitem_99, getitem_100, alias_19, getitem_102, getitem_103, getitem_104, 0, 0, 0.0, False, getitem_107, getitem_108)
    del alias_19
    del buf196
    del getitem_100
    del getitem_102
    del getitem_103
    del getitem_104
    del getitem_107
    del getitem_108
    del getitem_98
    del getitem_99
    buf200 = buf199[0]
    buf201 = buf199[1]
    buf202 = buf199[2]
    del buf199
    buf203 = buf175; del buf175  # reuse
    cpp_fused_clone_35(c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()))
    del buf200
    del buf201
    buf204 = reinterpret_tensor(buf202, (2056, 512), (512, 1), 0); del buf202  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf203, (2056, 1536), (1536, 1), 0), permute_219, out=buf204)
    del permute_219
    buf205 = empty((1536, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf203, (1536, 2056), (1, 1536), 0), view_65, out=buf205)
    del view_65
    buf206 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf207 = buf192; del buf192  # reuse
    buf208 = buf191; del buf191  # reuse
    buf209 = empty((512, ), device='cpu', dtype=torch.float32)
    buf210 = empty((512, ), device='cpu', dtype=torch.float32)
    buf211 = buf195; del buf195  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_36(c_void_p(buf211.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(mul_42.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()))
    del div_14
    del mul_42
    del primals_81
    buf212 = reinterpret_tensor(buf187, (2056, 2048), (2048, 1), 0); del buf187  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf211, (2056, 512), (512, 1), 0), permute_223, out=buf212)
    del permute_223
    buf213 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf211, (512, 2056), (1, 512), 0), view_63, out=buf213)
    del view_63
    buf214 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf215 = reinterpret_tensor(buf212, (8, 257, 2048), (526336, 2048, 1), 0); del buf212  # reuse
    cpp_fused_gelu_gelu_backward_sum_37(c_void_p(buf215.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf214.data_ptr()))
    del addmm_22
    buf216 = buf204; del buf204  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (2056, 2048), (2048, 1), 0), permute_227, out=buf216)
    del permute_227
    buf217 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (2048, 2056), (1, 2048), 0), view_61, out=buf217)
    del view_61
    buf218 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf219 = buf208; del buf208  # reuse
    buf220 = buf207; del buf207  # reuse
    buf221 = empty((512, ), device='cpu', dtype=torch.float32)
    buf222 = empty((512, ), device='cpu', dtype=torch.float32)
    buf223 = buf211; del buf211  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_38(c_void_p(buf223.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(mul_37.data_ptr()), c_void_p(div_15.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()))
    del div_15
    del mul_37
    del primals_75
    buf224 = buf216; del buf216  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (2056, 512), (512, 1), 0), permute_231, out=buf224)
    del permute_231
    buf225 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (512, 2056), (1, 512), 0), view_59, out=buf225)
    del view_59
    buf226 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_39(c_void_p(buf223.data_ptr()), c_void_p(buf226.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf227 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf224, (8, 8, 257, 64), (131584, 64, 512, 1), 0), getitem_82, getitem_83, getitem_84, alias_20, getitem_86, getitem_87, getitem_88, 0, 0, 0.0, False, getitem_91, getitem_92)
    del alias_20
    del buf224
    del getitem_82
    del getitem_83
    del getitem_84
    del getitem_86
    del getitem_87
    del getitem_88
    del getitem_91
    del getitem_92
    buf228 = buf227[0]
    buf229 = buf227[1]
    buf230 = buf227[2]
    del buf227
    buf231 = buf203; del buf203  # reuse
    cpp_fused_clone_40(c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del buf228
    del buf229
    buf232 = reinterpret_tensor(buf230, (2056, 512), (512, 1), 0); del buf230  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (2056, 1536), (1536, 1), 0), permute_237, out=buf232)
    del permute_237
    buf233 = empty((1536, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (1536, 2056), (1, 1536), 0), view_55, out=buf233)
    del view_55
    buf234 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf235 = buf220; del buf220  # reuse
    buf236 = buf219; del buf219  # reuse
    buf237 = empty((512, ), device='cpu', dtype=torch.float32)
    buf238 = empty((512, ), device='cpu', dtype=torch.float32)
    buf239 = buf223; del buf223  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_41(c_void_p(buf239.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    del div_16
    del mul_35
    del primals_69
    buf240 = reinterpret_tensor(buf215, (2056, 2048), (2048, 1), 0); del buf215  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf239, (2056, 512), (512, 1), 0), permute_241, out=buf240)
    del permute_241
    buf241 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf239, (512, 2056), (1, 512), 0), view_53, out=buf241)
    del view_53
    buf242 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf243 = reinterpret_tensor(buf240, (8, 257, 2048), (526336, 2048, 1), 0); del buf240  # reuse
    cpp_fused_gelu_gelu_backward_sum_42(c_void_p(buf243.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(addmm_18.data_ptr()), c_void_p(buf242.data_ptr()))
    del addmm_18
    buf244 = buf232; del buf232  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (2056, 2048), (2048, 1), 0), permute_245, out=buf244)
    del permute_245
    buf245 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (2048, 2056), (1, 2048), 0), view_51, out=buf245)
    del view_51
    buf246 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf247 = buf236; del buf236  # reuse
    buf248 = buf235; del buf235  # reuse
    buf249 = empty((512, ), device='cpu', dtype=torch.float32)
    buf250 = empty((512, ), device='cpu', dtype=torch.float32)
    buf251 = buf239; del buf239  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_43(c_void_p(buf251.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(mul_30.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()))
    del div_17
    del mul_30
    del primals_63
    buf252 = buf244; del buf244  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (2056, 512), (512, 1), 0), permute_249, out=buf252)
    del permute_249
    buf253 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (512, 2056), (1, 512), 0), view_49, out=buf253)
    del view_49
    buf254 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_44(c_void_p(buf251.data_ptr()), c_void_p(buf254.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf255 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf252, (8, 8, 257, 64), (131584, 64, 512, 1), 0), getitem_66, getitem_67, getitem_68, alias_21, getitem_70, getitem_71, getitem_72, 0, 0, 0.0, False, getitem_75, getitem_76)
    del alias_21
    del buf252
    del getitem_66
    del getitem_67
    del getitem_68
    del getitem_70
    del getitem_71
    del getitem_72
    del getitem_75
    del getitem_76
    buf256 = buf255[0]
    buf257 = buf255[1]
    buf258 = buf255[2]
    del buf255
    buf259 = buf231; del buf231  # reuse
    cpp_fused_clone_45(c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()))
    del buf256
    del buf257
    buf260 = reinterpret_tensor(buf258, (2056, 512), (512, 1), 0); del buf258  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf259, (2056, 1536), (1536, 1), 0), permute_255, out=buf260)
    del permute_255
    buf261 = empty((1536, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf259, (1536, 2056), (1, 1536), 0), view_45, out=buf261)
    del view_45
    buf262 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf263 = buf248; del buf248  # reuse
    buf264 = buf247; del buf247  # reuse
    buf265 = empty((512, ), device='cpu', dtype=torch.float32)
    buf266 = empty((512, ), device='cpu', dtype=torch.float32)
    buf267 = buf251; del buf251  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_46(c_void_p(buf267.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(mul_28.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    del div_18
    del mul_28
    del primals_57
    buf268 = reinterpret_tensor(buf243, (2056, 2048), (2048, 1), 0); del buf243  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf267, (2056, 512), (512, 1), 0), permute_259, out=buf268)
    del permute_259
    buf269 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf267, (512, 2056), (1, 512), 0), view_43, out=buf269)
    del view_43
    buf270 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf271 = reinterpret_tensor(buf268, (8, 257, 2048), (526336, 2048, 1), 0); del buf268  # reuse
    cpp_fused_gelu_gelu_backward_sum_47(c_void_p(buf271.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(buf270.data_ptr()))
    del addmm_14
    buf272 = buf260; del buf260  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf271, (2056, 2048), (2048, 1), 0), permute_263, out=buf272)
    del permute_263
    buf273 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf271, (2048, 2056), (1, 2048), 0), view_41, out=buf273)
    del view_41
    buf274 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf275 = buf264; del buf264  # reuse
    buf276 = buf263; del buf263  # reuse
    buf277 = empty((512, ), device='cpu', dtype=torch.float32)
    buf278 = empty((512, ), device='cpu', dtype=torch.float32)
    buf279 = buf267; del buf267  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_48(c_void_p(buf279.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(mul_23.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()))
    del buf271
    del div_19
    del mul_23
    del primals_51
    buf280 = buf272; del buf272  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (2056, 512), (512, 1), 0), permute_267, out=buf280)
    del permute_267
    buf281 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (512, 2056), (1, 512), 0), view_39, out=buf281)
    del view_39
    buf282 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_49(c_void_p(buf279.data_ptr()), c_void_p(buf282.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf283 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf280, (8, 8, 257, 64), (131584, 64, 512, 1), 0), getitem_50, getitem_51, getitem_52, alias_22, getitem_54, getitem_55, getitem_56, 0, 0, 0.0, False, getitem_59, getitem_60)
    del alias_22
    del buf280
    del getitem_50
    del getitem_51
    del getitem_52
    del getitem_54
    del getitem_55
    del getitem_56
    del getitem_59
    del getitem_60
    buf284 = buf283[0]
    buf285 = buf283[1]
    buf286 = buf283[2]
    del buf283
    buf287 = buf259; del buf259  # reuse
    cpp_fused_clone_50(c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    del buf284
    del buf285
    buf288 = reinterpret_tensor(buf286, (2056, 512), (512, 1), 0); del buf286  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (2056, 1536), (1536, 1), 0), permute_273, out=buf288)
    del permute_273
    buf289 = empty((1536, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (1536, 2056), (1, 1536), 0), view_35, out=buf289)
    del view_35
    buf290 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf291 = buf276; del buf276  # reuse
    buf292 = buf275; del buf275  # reuse
    buf293 = empty((512, ), device='cpu', dtype=torch.float32)
    buf294 = empty((512, ), device='cpu', dtype=torch.float32)
    buf295 = buf279; del buf279  # reuse
    buf296 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_51(c_void_p(buf295.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(cat_1.data_ptr()), c_void_p(getitem_49.data_ptr()), c_void_p(rsqrt_6.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()))
    del buf287
    del buf288
    del buf291
    del buf292
    del cat_1
    del getitem_49
    del primals_45
    del rsqrt_6
    buf297 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf295, (512, 8), (1, 131584), 0), view_32, out=buf297)
    del view_32
    buf298 = empty((8, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf295, (8, 512), (131584, 1), 0), permute_280, out=buf298)
    del permute_280
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf299 = aten.convolution_backward(reinterpret_tensor(buf295, (8, 512, 16, 16), (131584, 1, 8192, 512), 512), view_31, primals_41, [512], [2, 2], [1, 1], [1, 1], False, [0, 0], 256, [True, True, True])
    del buf295
    del primals_41
    del view_31
    buf300 = buf299[0]
    buf301 = buf299[1]
    buf302 = buf299[2]
    del buf299
    buf303 = empty((8, 962, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_slice_backward_52(c_void_p(buf300.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf303.data_ptr()))
    del buf298
    del buf300
    buf304 = empty((7696, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf303, (7696, 256), (256, 1), 0), permute_283, out=buf304)
    del permute_283
    buf305 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf303, (256, 7696), (1, 256), 0), view_29, out=buf305)
    del view_29
    buf306 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf307 = reinterpret_tensor(buf304, (8, 962, 1024), (985088, 1024, 1), 0); del buf304  # reuse
    cpp_fused_gelu_gelu_backward_sum_53(c_void_p(buf307.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf306.data_ptr()))
    del addmm_10
    buf308 = empty((7696, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf307, (7696, 1024), (1024, 1), 0), permute_287, out=buf308)
    del permute_287
    buf309 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf307, (1024, 7696), (1, 1024), 0), view_27, out=buf309)
    del view_27
    buf310 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf311 = empty_strided((8, 962, 1), (962, 1, 7696), device='cpu', dtype=torch.float32)
    buf312 = empty_strided((8, 962, 1), (962, 1, 7696), device='cpu', dtype=torch.float32)
    buf313 = empty((256, ), device='cpu', dtype=torch.float32)
    buf314 = empty((256, ), device='cpu', dtype=torch.float32)
    buf315 = buf303; del buf303  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_54(c_void_p(buf315.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()))
    del div_21
    del mul_16
    del primals_35
    buf316 = buf308; del buf308  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf315, (7696, 256), (256, 1), 0), permute_291, out=buf316)
    del permute_291
    buf317 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf315, (256, 7696), (1, 256), 0), view_25, out=buf317)
    del view_25
    buf318 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_55(c_void_p(buf315.data_ptr()), c_void_p(buf318.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf319 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf316, (8, 4, 962, 64), (246272, 64, 256, 1), 0), getitem_34, getitem_35, getitem_36, alias_23, getitem_38, getitem_39, getitem_40, 0, 0, 0.0, False, getitem_43, getitem_44)
    del alias_23
    del buf316
    del getitem_34
    del getitem_35
    del getitem_36
    del getitem_38
    del getitem_39
    del getitem_40
    del getitem_43
    del getitem_44
    buf320 = buf319[0]
    buf321 = buf319[1]
    buf322 = buf319[2]
    del buf319
    buf323 = empty((8, 962, 3, 4, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_56(c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()))
    del buf320
    del buf321
    buf324 = reinterpret_tensor(buf322, (7696, 256), (256, 1), 0); del buf322  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf323, (7696, 768), (768, 1), 0), permute_297, out=buf324)
    del permute_297
    buf325 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf323, (768, 7696), (1, 768), 0), view_21, out=buf325)
    del view_21
    buf326 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf327 = buf312; del buf312  # reuse
    buf328 = buf311; del buf311  # reuse
    buf329 = empty((256, ), device='cpu', dtype=torch.float32)
    buf330 = empty((256, ), device='cpu', dtype=torch.float32)
    buf331 = buf315; del buf315  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_57(c_void_p(buf331.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(mul_14.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()))
    del div_22
    del mul_14
    del primals_29
    buf332 = reinterpret_tensor(buf307, (7696, 1024), (1024, 1), 0); del buf307  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (7696, 256), (256, 1), 0), permute_301, out=buf332)
    del permute_301
    buf333 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (256, 7696), (1, 256), 0), view_19, out=buf333)
    del view_19
    buf334 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf335 = reinterpret_tensor(buf332, (8, 962, 1024), (985088, 1024, 1), 0); del buf332  # reuse
    cpp_fused_gelu_gelu_backward_sum_58(c_void_p(buf335.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(buf334.data_ptr()))
    del addmm_6
    buf336 = buf324; del buf324  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf335, (7696, 1024), (1024, 1), 0), permute_305, out=buf336)
    del permute_305
    buf337 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf335, (1024, 7696), (1, 1024), 0), view_17, out=buf337)
    del view_17
    buf338 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf339 = buf328; del buf328  # reuse
    buf340 = buf327; del buf327  # reuse
    buf341 = empty((256, ), device='cpu', dtype=torch.float32)
    buf342 = empty((256, ), device='cpu', dtype=torch.float32)
    buf343 = buf331; del buf331  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_59(c_void_p(buf343.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(mul_9.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    del div_23
    del mul_9
    del primals_23
    buf344 = buf336; del buf336  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf343, (7696, 256), (256, 1), 0), permute_309, out=buf344)
    del permute_309
    buf345 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf343, (256, 7696), (1, 256), 0), view_15, out=buf345)
    del view_15
    buf346 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_60(c_void_p(buf343.data_ptr()), c_void_p(buf346.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf347 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf344, (8, 4, 962, 64), (246272, 64, 256, 1), 0), getitem_18, getitem_19, getitem_20, alias_24, getitem_22, getitem_23, getitem_24, 0, 0, 0.0, False, getitem_27, getitem_28)
    del alias_24
    del buf344
    del getitem_18
    del getitem_19
    del getitem_20
    del getitem_22
    del getitem_23
    del getitem_24
    del getitem_27
    del getitem_28
    buf348 = buf347[0]
    buf349 = buf347[1]
    buf350 = buf347[2]
    del buf347
    buf351 = buf323; del buf323  # reuse
    cpp_fused_clone_61(c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()))
    del buf348
    del buf349
    buf352 = reinterpret_tensor(buf350, (7696, 256), (256, 1), 0); del buf350  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf351, (7696, 768), (768, 1), 0), permute_315, out=buf352)
    del permute_315
    buf353 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf351, (768, 7696), (1, 768), 0), view_11, out=buf353)
    del view_11
    buf354 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf355 = buf340; del buf340  # reuse
    buf356 = buf339; del buf339  # reuse
    buf357 = empty((256, ), device='cpu', dtype=torch.float32)
    buf358 = empty((256, ), device='cpu', dtype=torch.float32)
    buf359 = buf343; del buf343  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_62(c_void_p(buf359.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(mul_7.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()))
    del div_24
    del mul_7
    del primals_17
    buf360 = reinterpret_tensor(buf335, (7696, 1024), (1024, 1), 0); del buf335  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf359, (7696, 256), (256, 1), 0), permute_319, out=buf360)
    del permute_319
    buf361 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf359, (256, 7696), (1, 256), 0), view_9, out=buf361)
    del view_9
    buf362 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf363 = reinterpret_tensor(buf360, (8, 962, 1024), (985088, 1024, 1), 0); del buf360  # reuse
    cpp_fused_gelu_gelu_backward_sum_63(c_void_p(buf363.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(buf362.data_ptr()))
    del addmm_2
    buf364 = buf352; del buf352  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (7696, 1024), (1024, 1), 0), permute_323, out=buf364)
    del permute_323
    buf365 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (1024, 7696), (1, 1024), 0), view_7, out=buf365)
    del view_7
    buf366 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf367 = buf356; del buf356  # reuse
    buf368 = buf355; del buf355  # reuse
    buf369 = empty((256, ), device='cpu', dtype=torch.float32)
    buf370 = empty((256, ), device='cpu', dtype=torch.float32)
    buf371 = buf359; del buf359  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_64(c_void_p(buf371.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(mul_2.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()))
    del buf363
    del div_25
    del mul_2
    del primals_11
    buf372 = buf364; del buf364  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (7696, 256), (256, 1), 0), permute_327, out=buf372)
    del permute_327
    buf373 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (256, 7696), (1, 256), 0), view_5, out=buf373)
    del view_5
    buf374 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_65(c_void_p(buf371.data_ptr()), c_void_p(buf374.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf375 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf372, (8, 4, 962, 64), (246272, 64, 256, 1), 0), getitem_2, getitem_3, getitem_4, alias_25, getitem_6, getitem_7, getitem_8, 0, 0, 0.0, False, getitem_11, getitem_12)
    del alias_25
    del buf372
    del getitem_11
    del getitem_12
    del getitem_2
    del getitem_3
    del getitem_4
    del getitem_6
    del getitem_7
    del getitem_8
    buf376 = buf375[0]
    buf377 = buf375[1]
    buf378 = buf375[2]
    del buf375
    buf379 = buf351; del buf351  # reuse
    cpp_fused_clone_66(c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()))
    del buf376
    del buf377
    buf380 = reinterpret_tensor(buf378, (7696, 256), (256, 1), 0); del buf378  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf379, (7696, 768), (768, 1), 0), permute_333, out=buf380)
    del permute_333
    buf381 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf379, (768, 7696), (1, 768), 0), view_1, out=buf381)
    del view_1
    buf382 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf383 = buf368; del buf368  # reuse
    buf384 = buf367; del buf367  # reuse
    buf385 = empty((256, ), device='cpu', dtype=torch.float32)
    buf386 = empty((256, ), device='cpu', dtype=torch.float32)
    buf387 = buf371; del buf371  # reuse
    buf388 = empty((1, 1, 256), device='cpu', dtype=torch.float32)
    buf389 = empty((1, 256, 31, 31), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_67(c_void_p(buf387.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(cat.data_ptr()), c_void_p(getitem_1.data_ptr()), c_void_p(rsqrt.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()))
    del buf379
    del buf380
    del buf383
    del buf384
    del cat
    del getitem_1
    del primals_5
    del rsqrt
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf390 = aten.convolution_backward(reinterpret_tensor(buf387, (8, 256, 31, 31), (246272, 1, 7936, 256), 256), primals_173, primals_3, [256], [7, 7], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf387
    del primals_173
    del primals_3
    buf391 = buf390[1]
    buf392 = buf390[2]
    return (buf389, buf388, buf391, buf392, buf385, buf386, reinterpret_tensor(buf381, (768, 256), (256, 1), 0), reinterpret_tensor(buf382, (768, ), (1, ), 0), reinterpret_tensor(buf373, (256, 256), (256, 1), 0), reinterpret_tensor(buf374, (256, ), (1, ), 0), buf369, buf370, reinterpret_tensor(buf365, (1024, 256), (256, 1), 0), reinterpret_tensor(buf366, (1024, ), (1, ), 0), reinterpret_tensor(buf361, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf362, (256, ), (1, ), 0), buf357, buf358, reinterpret_tensor(buf353, (768, 256), (256, 1), 0), reinterpret_tensor(buf354, (768, ), (1, ), 0), reinterpret_tensor(buf345, (256, 256), (256, 1), 0), reinterpret_tensor(buf346, (256, ), (1, ), 0), buf341, buf342, reinterpret_tensor(buf337, (1024, 256), (256, 1), 0), reinterpret_tensor(buf338, (1024, ), (1, ), 0), reinterpret_tensor(buf333, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf334, (256, ), (1, ), 0), buf329, buf330, reinterpret_tensor(buf325, (768, 256), (256, 1), 0), reinterpret_tensor(buf326, (768, ), (1, ), 0), reinterpret_tensor(buf317, (256, 256), (256, 1), 0), reinterpret_tensor(buf318, (256, ), (1, ), 0), buf313, buf314, reinterpret_tensor(buf309, (1024, 256), (256, 1), 0), reinterpret_tensor(buf310, (1024, ), (1, ), 0), reinterpret_tensor(buf305, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf306, (256, ), (1, ), 0), buf301, buf302, reinterpret_tensor(buf297, (512, 256), (256, 1), 0), reinterpret_tensor(buf296, (512, ), (1, ), 0), buf293, buf294, reinterpret_tensor(buf289, (1536, 512), (512, 1), 0), reinterpret_tensor(buf290, (1536, ), (1, ), 0), reinterpret_tensor(buf281, (512, 512), (512, 1), 0), reinterpret_tensor(buf282, (512, ), (1, ), 0), buf277, buf278, reinterpret_tensor(buf273, (2048, 512), (512, 1), 0), reinterpret_tensor(buf274, (2048, ), (1, ), 0), reinterpret_tensor(buf269, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf270, (512, ), (1, ), 0), buf265, buf266, reinterpret_tensor(buf261, (1536, 512), (512, 1), 0), reinterpret_tensor(buf262, (1536, ), (1, ), 0), reinterpret_tensor(buf253, (512, 512), (512, 1), 0), reinterpret_tensor(buf254, (512, ), (1, ), 0), buf249, buf250, reinterpret_tensor(buf245, (2048, 512), (512, 1), 0), reinterpret_tensor(buf246, (2048, ), (1, ), 0), reinterpret_tensor(buf241, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf242, (512, ), (1, ), 0), buf237, buf238, reinterpret_tensor(buf233, (1536, 512), (512, 1), 0), reinterpret_tensor(buf234, (1536, ), (1, ), 0), reinterpret_tensor(buf225, (512, 512), (512, 1), 0), reinterpret_tensor(buf226, (512, ), (1, ), 0), buf221, buf222, reinterpret_tensor(buf217, (2048, 512), (512, 1), 0), reinterpret_tensor(buf218, (2048, ), (1, ), 0), reinterpret_tensor(buf213, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf214, (512, ), (1, ), 0), buf209, buf210, reinterpret_tensor(buf205, (1536, 512), (512, 1), 0), reinterpret_tensor(buf206, (1536, ), (1, ), 0), reinterpret_tensor(buf197, (512, 512), (512, 1), 0), reinterpret_tensor(buf198, (512, ), (1, ), 0), buf193, buf194, reinterpret_tensor(buf189, (2048, 512), (512, 1), 0), reinterpret_tensor(buf190, (2048, ), (1, ), 0), reinterpret_tensor(buf185, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf186, (512, ), (1, ), 0), buf181, buf182, reinterpret_tensor(buf177, (1536, 512), (512, 1), 0), reinterpret_tensor(buf178, (1536, ), (1, ), 0), reinterpret_tensor(buf169, (512, 512), (512, 1), 0), reinterpret_tensor(buf170, (512, ), (1, ), 0), buf165, buf166, reinterpret_tensor(buf161, (2048, 512), (512, 1), 0), reinterpret_tensor(buf162, (2048, ), (1, ), 0), reinterpret_tensor(buf157, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf158, (512, ), (1, ), 0), buf153, buf154, reinterpret_tensor(buf149, (1536, 512), (512, 1), 0), reinterpret_tensor(buf150, (1536, ), (1, ), 0), reinterpret_tensor(buf141, (512, 512), (512, 1), 0), reinterpret_tensor(buf142, (512, ), (1, ), 0), buf137, buf138, reinterpret_tensor(buf133, (2048, 512), (512, 1), 0), reinterpret_tensor(buf134, (2048, ), (1, ), 0), reinterpret_tensor(buf129, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf130, (512, ), (1, ), 0), buf125, buf126, reinterpret_tensor(buf121, (1024, 512), (512, 1), 0), reinterpret_tensor(buf120, (1024, ), (1, ), 0), buf117, buf118, reinterpret_tensor(buf113, (3072, 1024), (1024, 1), 0), reinterpret_tensor(buf114, (3072, ), (1, ), 0), reinterpret_tensor(buf105, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf106, (1024, ), (1, ), 0), buf101, buf102, reinterpret_tensor(buf97, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf98, (4096, ), (1, ), 0), reinterpret_tensor(buf93, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf94, (1024, ), (1, ), 0), buf89, buf90, reinterpret_tensor(buf85, (3072, 1024), (1024, 1), 0), reinterpret_tensor(buf86, (3072, ), (1, ), 0), reinterpret_tensor(buf77, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf78, (1024, ), (1, ), 0), buf73, buf74, reinterpret_tensor(buf69, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf70, (4096, ), (1, ), 0), reinterpret_tensor(buf65, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf66, (1024, ), (1, ), 0), buf61, buf62, reinterpret_tensor(buf57, (3072, 1024), (1024, 1), 0), reinterpret_tensor(buf58, (3072, ), (1, ), 0), reinterpret_tensor(buf49, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf50, (1024, ), (1, ), 0), buf45, buf46, reinterpret_tensor(buf41, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf42, (4096, ), (1, ), 0), reinterpret_tensor(buf37, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf38, (1024, ), (1, ), 0), buf33, buf34, reinterpret_tensor(buf29, (3072, 1024), (1024, 1), 0), reinterpret_tensor(buf30, (3072, ), (1, ), 0), reinterpret_tensor(buf21, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf22, (1024, ), (1, ), 0), buf17, buf18, reinterpret_tensor(buf13, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf14, (4096, ), (1, ), 0), reinterpret_tensor(buf9, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf10, (1024, ), (1, ), 0), buf5, buf6, reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((256, 3, 14, 14), (588, 1, 42, 3), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((1024, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cat = rand_strided((8, 962, 256), (246272, 256, 1), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((8, 962, 1), (962, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt = rand_strided((8, 962, 1), (962, 1, 1), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((7696, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_4 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_6 = rand_strided((8, 4, 962), (3848, 1, 4), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_8 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_11 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_12 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_5 = rand_strided((7696, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_2 = rand_strided((8, 962, 256), (246272, 256, 1), device='cpu', dtype=torch.float32)
    view_7 = rand_strided((7696, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((7696, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_9 = rand_strided((7696, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    mul_7 = rand_strided((8, 962, 256), (246272, 256, 1), device='cpu', dtype=torch.float32)
    view_11 = rand_strided((7696, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_18 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_19 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_20 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_22 = rand_strided((8, 4, 962), (3848, 1, 4), device='cpu', dtype=torch.float32)
    getitem_23 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_24 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_27 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_28 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_15 = rand_strided((7696, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_9 = rand_strided((8, 962, 256), (246272, 256, 1), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((7696, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_6 = rand_strided((7696, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_19 = rand_strided((7696, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    mul_14 = rand_strided((8, 962, 256), (246272, 256, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((7696, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_34 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_35 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_36 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_38 = rand_strided((8, 4, 962), (3848, 1, 4), device='cpu', dtype=torch.float32)
    getitem_39 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_40 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_43 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_44 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_25 = rand_strided((7696, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((8, 962, 256), (246272, 256, 1), device='cpu', dtype=torch.float32)
    view_27 = rand_strided((7696, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((7696, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_29 = rand_strided((7696, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_31 = rand_strided((8, 256, 31, 31), (246272, 1, 7936, 256), device='cpu', dtype=torch.float32)
    view_32 = rand_strided((8, 256), (246272, 1), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((8, 257, 512), (131584, 512, 1), device='cpu', dtype=torch.float32)
    getitem_49 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_6 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    view_35 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_50 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_52 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_54 = rand_strided((8, 8, 257), (2056, 1, 8), device='cpu', dtype=torch.float32)
    getitem_55 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_56 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_59 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_60 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_39 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    mul_23 = rand_strided((8, 257, 512), (131584, 512, 1), device='cpu', dtype=torch.float32)
    view_41 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((2056, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_43 = rand_strided((2056, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_28 = rand_strided((8, 257, 512), (131584, 512, 1), device='cpu', dtype=torch.float32)
    view_45 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_66 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_68 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_70 = rand_strided((8, 8, 257), (2056, 1, 8), device='cpu', dtype=torch.float32)
    getitem_71 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_72 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_75 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_76 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_49 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    mul_30 = rand_strided((8, 257, 512), (131584, 512, 1), device='cpu', dtype=torch.float32)
    view_51 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_18 = rand_strided((2056, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_53 = rand_strided((2056, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_35 = rand_strided((8, 257, 512), (131584, 512, 1), device='cpu', dtype=torch.float32)
    view_55 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_82 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_83 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_84 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_86 = rand_strided((8, 8, 257), (2056, 1, 8), device='cpu', dtype=torch.float32)
    getitem_87 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_88 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_91 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_92 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_59 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    mul_37 = rand_strided((8, 257, 512), (131584, 512, 1), device='cpu', dtype=torch.float32)
    view_61 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((2056, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_63 = rand_strided((2056, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_42 = rand_strided((8, 257, 512), (131584, 512, 1), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_98 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_99 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_100 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_102 = rand_strided((8, 8, 257), (2056, 1, 8), device='cpu', dtype=torch.float32)
    getitem_103 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_104 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_107 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_108 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_69 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    mul_44 = rand_strided((8, 257, 512), (131584, 512, 1), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_26 = rand_strided((2056, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((2056, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_49 = rand_strided((8, 257, 512), (131584, 512, 1), device='cpu', dtype=torch.float32)
    view_75 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_114 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_115 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_116 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_118 = rand_strided((8, 8, 257), (2056, 1, 8), device='cpu', dtype=torch.float32)
    getitem_119 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_120 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_123 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_124 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_79 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    mul_51 = rand_strided((8, 257, 512), (131584, 512, 1), device='cpu', dtype=torch.float32)
    view_81 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_30 = rand_strided((2056, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_83 = rand_strided((2056, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_56 = rand_strided((8, 257, 512), (131584, 512, 1), device='cpu', dtype=torch.float32)
    view_85 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_130 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_131 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_132 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_134 = rand_strided((8, 8, 257), (2056, 1, 8), device='cpu', dtype=torch.float32)
    getitem_135 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_136 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_139 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_140 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_89 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    mul_58 = rand_strided((8, 257, 512), (131584, 512, 1), device='cpu', dtype=torch.float32)
    view_91 = rand_strided((2056, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((2056, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_93 = rand_strided((2056, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_95 = rand_strided((8, 512, 16, 16), (131584, 1, 8192, 512), device='cpu', dtype=torch.float32)
    view_96 = rand_strided((8, 512), (131584, 1), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cpu', dtype=torch.float32)
    getitem_145 = rand_strided((8, 65, 1), (65, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_18 = rand_strided((8, 65, 1), (65, 1, 1), device='cpu', dtype=torch.float32)
    view_99 = rand_strided((520, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_146 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_147 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_148 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_150 = rand_strided((8, 16, 65), (1040, 1, 16), device='cpu', dtype=torch.float32)
    getitem_151 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_152 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_155 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_156 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_103 = rand_strided((520, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    mul_65 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cpu', dtype=torch.float32)
    view_105 = rand_strided((520, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_38 = rand_strided((520, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_107 = rand_strided((520, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_70 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cpu', dtype=torch.float32)
    view_109 = rand_strided((520, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_162 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_163 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_164 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_166 = rand_strided((8, 16, 65), (1040, 1, 16), device='cpu', dtype=torch.float32)
    getitem_167 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_168 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_171 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_172 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_113 = rand_strided((520, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    mul_72 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cpu', dtype=torch.float32)
    view_115 = rand_strided((520, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_42 = rand_strided((520, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_117 = rand_strided((520, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_77 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cpu', dtype=torch.float32)
    view_119 = rand_strided((520, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_178 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_179 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_180 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_182 = rand_strided((8, 16, 65), (1040, 1, 16), device='cpu', dtype=torch.float32)
    getitem_183 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_184 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_187 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_188 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_123 = rand_strided((520, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    mul_79 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cpu', dtype=torch.float32)
    view_125 = rand_strided((520, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((520, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_127 = rand_strided((520, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_84 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cpu', dtype=torch.float32)
    view_129 = rand_strided((520, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_194 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_195 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_196 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_198 = rand_strided((8, 16, 65), (1040, 1, 16), device='cpu', dtype=torch.float32)
    getitem_199 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_200 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_203 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_204 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_133 = rand_strided((520, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    mul_86 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cpu', dtype=torch.float32)
    view_135 = rand_strided((520, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_50 = rand_strided((520, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_137 = rand_strided((520, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_91 = rand_strided((8, 1, 1024), (1024, 1024, 1), device='cpu', dtype=torch.float32)
    clone_41 = rand_strided((8, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_87 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_91 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_95 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((8, 65, 1), (65, 1, 1), device='cpu', dtype=torch.float32)
    permute_99 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    alias_13 = rand_strided((8, 16, 65, 64), (66560, 1, 1024, 16), device='cpu', dtype=torch.float32)
    permute_105 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((8, 65, 1), (65, 1, 1), device='cpu', dtype=torch.float32)
    permute_109 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_113 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((8, 65, 1), (65, 1, 1), device='cpu', dtype=torch.float32)
    permute_117 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    alias_14 = rand_strided((8, 16, 65, 64), (66560, 1, 1024, 16), device='cpu', dtype=torch.float32)
    permute_123 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((8, 65, 1), (65, 1, 1), device='cpu', dtype=torch.float32)
    permute_127 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_131 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((8, 65, 1), (65, 1, 1), device='cpu', dtype=torch.float32)
    permute_135 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    alias_15 = rand_strided((8, 16, 65, 64), (66560, 1, 1024, 16), device='cpu', dtype=torch.float32)
    permute_141 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((8, 65, 1), (65, 1, 1), device='cpu', dtype=torch.float32)
    permute_145 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_149 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_7 = rand_strided((8, 65, 1), (65, 1, 1), device='cpu', dtype=torch.float32)
    permute_153 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    alias_16 = rand_strided((8, 16, 65, 64), (66560, 1, 1024, 16), device='cpu', dtype=torch.float32)
    permute_159 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_166 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_169 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_173 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    permute_177 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    alias_17 = rand_strided((8, 8, 257, 64), (131584, 1, 512, 8), device='cpu', dtype=torch.float32)
    permute_183 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_10 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    permute_187 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_191 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    permute_195 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    alias_18 = rand_strided((8, 8, 257, 64), (131584, 1, 512, 8), device='cpu', dtype=torch.float32)
    permute_201 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    permute_205 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_209 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    permute_213 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    alias_19 = rand_strided((8, 8, 257, 64), (131584, 1, 512, 8), device='cpu', dtype=torch.float32)
    permute_219 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    permute_223 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_227 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    permute_231 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    alias_20 = rand_strided((8, 8, 257, 64), (131584, 1, 512, 8), device='cpu', dtype=torch.float32)
    permute_237 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    permute_241 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_245 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    permute_249 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    alias_21 = rand_strided((8, 8, 257, 64), (131584, 1, 512, 8), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    permute_259 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_263 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((8, 257, 1), (257, 1, 1), device='cpu', dtype=torch.float32)
    permute_267 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    alias_22 = rand_strided((8, 8, 257, 64), (131584, 1, 512, 8), device='cpu', dtype=torch.float32)
    permute_273 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_280 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_283 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_287 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((8, 962, 1), (962, 1, 1), device='cpu', dtype=torch.float32)
    permute_291 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    alias_23 = rand_strided((8, 4, 962, 64), (246272, 1, 256, 4), device='cpu', dtype=torch.float32)
    permute_297 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((8, 962, 1), (962, 1, 1), device='cpu', dtype=torch.float32)
    permute_301 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_305 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((8, 962, 1), (962, 1, 1), device='cpu', dtype=torch.float32)
    permute_309 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    alias_24 = rand_strided((8, 4, 962, 64), (246272, 1, 256, 4), device='cpu', dtype=torch.float32)
    permute_315 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((8, 962, 1), (962, 1, 1), device='cpu', dtype=torch.float32)
    permute_319 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_323 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((8, 962, 1), (962, 1, 1), device='cpu', dtype=torch.float32)
    permute_327 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    alias_25 = rand_strided((8, 4, 962, 64), (246272, 1, 256, 4), device='cpu', dtype=torch.float32)
    permute_333 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_121, primals_127, primals_133, primals_139, primals_145, primals_151, primals_157, primals_163, primals_169, primals_173, cat, getitem_1, rsqrt, view_1, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, getitem_11, getitem_12, view_5, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_18, getitem_19, getitem_20, getitem_22, getitem_23, getitem_24, getitem_27, getitem_28, view_15, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_34, getitem_35, getitem_36, getitem_38, getitem_39, getitem_40, getitem_43, getitem_44, view_25, mul_16, view_27, addmm_10, view_29, view_31, view_32, cat_1, getitem_49, rsqrt_6, view_35, getitem_50, getitem_51, getitem_52, getitem_54, getitem_55, getitem_56, getitem_59, getitem_60, view_39, mul_23, view_41, addmm_14, view_43, mul_28, view_45, getitem_66, getitem_67, getitem_68, getitem_70, getitem_71, getitem_72, getitem_75, getitem_76, view_49, mul_30, view_51, addmm_18, view_53, mul_35, view_55, getitem_82, getitem_83, getitem_84, getitem_86, getitem_87, getitem_88, getitem_91, getitem_92, view_59, mul_37, view_61, addmm_22, view_63, mul_42, view_65, getitem_98, getitem_99, getitem_100, getitem_102, getitem_103, getitem_104, getitem_107, getitem_108, view_69, mul_44, view_71, addmm_26, view_73, mul_49, view_75, getitem_114, getitem_115, getitem_116, getitem_118, getitem_119, getitem_120, getitem_123, getitem_124, view_79, mul_51, view_81, addmm_30, view_83, mul_56, view_85, getitem_130, getitem_131, getitem_132, getitem_134, getitem_135, getitem_136, getitem_139, getitem_140, view_89, mul_58, view_91, addmm_34, view_93, view_95, view_96, cat_2, getitem_145, rsqrt_18, view_99, getitem_146, getitem_147, getitem_148, getitem_150, getitem_151, getitem_152, getitem_155, getitem_156, view_103, mul_65, view_105, addmm_38, view_107, mul_70, view_109, getitem_162, getitem_163, getitem_164, getitem_166, getitem_167, getitem_168, getitem_171, getitem_172, view_113, mul_72, view_115, addmm_42, view_117, mul_77, view_119, getitem_178, getitem_179, getitem_180, getitem_182, getitem_183, getitem_184, getitem_187, getitem_188, view_123, mul_79, view_125, addmm_46, view_127, mul_84, view_129, getitem_194, getitem_195, getitem_196, getitem_198, getitem_199, getitem_200, getitem_203, getitem_204, view_133, mul_86, view_135, addmm_50, view_137, mul_91, clone_41, permute_87, div, permute_91, permute_95, div_1, permute_99, alias_13, permute_105, div_2, permute_109, permute_113, div_3, permute_117, alias_14, permute_123, div_4, permute_127, permute_131, div_5, permute_135, alias_15, permute_141, div_6, permute_145, permute_149, div_7, permute_153, alias_16, permute_159, permute_166, permute_169, permute_173, div_9, permute_177, alias_17, permute_183, div_10, permute_187, permute_191, div_11, permute_195, alias_18, permute_201, div_12, permute_205, permute_209, div_13, permute_213, alias_19, permute_219, div_14, permute_223, permute_227, div_15, permute_231, alias_20, permute_237, div_16, permute_241, permute_245, div_17, permute_249, alias_21, permute_255, div_18, permute_259, permute_263, div_19, permute_267, alias_22, permute_273, permute_280, permute_283, permute_287, div_21, permute_291, alias_23, permute_297, div_22, permute_301, permute_305, div_23, permute_309, alias_24, permute_315, div_24, permute_319, permute_323, div_25, permute_327, alias_25, permute_333, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pit_b_224', benchmark_compiled_module)
