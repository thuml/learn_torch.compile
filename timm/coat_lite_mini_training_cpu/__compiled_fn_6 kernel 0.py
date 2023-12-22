
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (512L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x2)];
                            auto tmp8 = in_ptr3[static_cast<long>(x2 + (512L*x1) + (25600L*x0))];
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
                        out_ptr1[static_cast<long>(x1 + (50L*x0))] = tmp_acc0;
                        out_ptr2[static_cast<long>(x1 + (50L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x1 + (50L*x0))];
                        auto tmp4 = in_ptr1[static_cast<long>(x2 + (512L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2)];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (50L*x0))];
                        auto tmp13 = in_ptr3[static_cast<long>(x2 + (512L*x1) + (25600L*x0))];
                        auto tmp14 = out_ptr2[static_cast<long>(x1 + (50L*x0))];
                        auto tmp1 = c10::convert<int>(x1);
                        auto tmp2 = static_cast<int>(0);
                        auto tmp3 = tmp1 == tmp2;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp9 = static_cast<float>(512.0);
                        auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                        auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp16 = decltype(tmp12)(tmp12 - tmp15);
                        auto tmp17 = decltype(tmp0)(tmp0 * tmp16);
                        out_ptr3[static_cast<long>(x2 + (512L*x1) + (25600L*x0))] = tmp17;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                        {
                            auto tmp3 = in_ptr1[static_cast<long>(x0 + (512L*x1))];
                            auto tmp6 = in_ptr3[static_cast<long>(x0 + (512L*x2) + (25600L*x1))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(819200L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_convolution_backward_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(320L + x2 + (1536L*x1) + (76800L*x0)));
                            auto tmp0 = c10::convert<int>(1L + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = masked_load(in_ptr1 + static_cast<long>(832L + x2 + (512L*x1) + (25600L*x0)), to_float_mask(tmp2));
                                return tmp4;
                            }
                            ;
                            auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                            auto tmp7 = tmp5 * tmp6;
                            tmp7.store(out_ptr1 + static_cast<long>(x2 + (192L*x1) + (9408L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (1536L*x1) + (76800L*x0)));
                    auto tmp0 = c10::convert<int>(1L + x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = masked_load(in_ptr0 + static_cast<long>(640L + x2 + (512L*x1) + (25600L*x0)), to_float_mask(tmp2));
                        return tmp4;
                    }
                    ;
                    auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp7 = tmp5 * tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x2 + (192L*x1) + (9408L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                {
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x1) + (76800L*x0)));
                    auto tmp0 = c10::convert<int>(1L + x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = masked_load(in_ptr0 + static_cast<long>(512L + x2 + (512L*x1) + (25600L*x0)), to_float_mask(tmp2));
                        return tmp4;
                    }
                    ;
                    auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp7 = tmp5 * tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (6272L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25600L*x0)));
                            auto tmp1 = static_cast<float>(0.125);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_stack_view_7 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (50L*x2) + (3200L*x1) + (25600L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x2) + (512L*x3) + (512L*x3_inner) + (25600L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(50L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (50L*x2) + (3200L*x1) + (25600L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x2) + (512L*x3) + (25600L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (64L*x1) + (512L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x0);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x2);
                                auto tmp7 = static_cast<long>(1);
                                auto tmp8 = tmp6 >= tmp7;
                                auto tmp9 = [&]
                                {
                                    auto tmp10 = c10::convert<long>(x2);
                                    auto tmp11 = static_cast<long>(0);
                                    auto tmp12 = tmp10 >= tmp11;
                                    auto tmp13 = [&]
                                    {
                                        auto tmp14 = in_ptr2[static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25600L*x0))];
                                        return tmp14;
                                    }
                                    ;
                                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                    auto tmp16 = in_ptr3[static_cast<long>(x3 + (64L*x1) + (512L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (25088L*x0))];
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                                auto tmp19 = static_cast<float>(0.0);
                                auto tmp20 = tmp8 ? tmp18 : tmp19;
                                auto tmp21 = in_ptr4[static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0))];
                                auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp24 = tmp0 >= tmp3;
                            auto tmp25 = static_cast<long>(16);
                            auto tmp26 = tmp0 < tmp25;
                            auto tmp27 = tmp24 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr0[static_cast<long>((-204800L) + x2 + (50L*x3) + (3200L*x1) + (25600L*x0))];
                                auto tmp30 = in_ptr1[static_cast<long>((-204800L) + x1 + (8L*x3) + (512L*x2) + (25600L*x0))];
                                auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                                auto tmp32 = out_ptr0[static_cast<long>((-4096L) + x3 + (64L*x1) + (512L*x0))];
                                auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                                auto tmp34 = decltype(tmp31)(tmp31 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp36 = tmp0 >= tmp25;
                            auto tmp37 = static_cast<long>(24);
                            auto tmp38 = tmp0 < tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x2);
                                auto tmp41 = static_cast<long>(1);
                                auto tmp42 = tmp40 >= tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = c10::convert<long>(x3 + (64L*x1));
                                    auto tmp45 = static_cast<long>(0);
                                    auto tmp46 = tmp44 >= tmp45;
                                    auto tmp47 = static_cast<long>(128);
                                    auto tmp48 = tmp44 < tmp47;
                                    auto tmp49 = [&]
                                    {
                                        auto tmp50 = in_ptr5[static_cast<long>((-100352L) + x3 + (64L*x1) + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (6272L*x0))];
                                        return tmp50;
                                    }
                                    ;
                                    auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                    auto tmp52 = tmp44 >= tmp47;
                                    auto tmp53 = static_cast<long>(320);
                                    auto tmp54 = tmp44 < tmp53;
                                    auto tmp55 = tmp52 & tmp54;
                                    auto tmp56 = [&]
                                    {
                                        auto tmp57 = in_ptr6[static_cast<long>((-150656L) + x3 + (64L*x1) + (192L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (9408L*x0))];
                                        return tmp57;
                                    }
                                    ;
                                    auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                                    auto tmp59 = tmp44 >= tmp53;
                                    auto tmp60 = static_cast<long>(512);
                                    auto tmp61 = tmp44 < tmp60;
                                    auto tmp62 = [&]
                                    {
                                        auto tmp63 = in_ptr7[static_cast<long>((-150848L) + x3 + (64L*x1) + (192L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (9408L*x0))];
                                        return tmp63;
                                    }
                                    ;
                                    auto tmp64 = tmp59 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                                    auto tmp65 = tmp55 ? tmp58 : tmp64;
                                    auto tmp66 = tmp48 ? tmp51 : tmp65;
                                    return tmp66;
                                }
                                ;
                                auto tmp67 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp68 = static_cast<float>(0.0);
                                auto tmp69 = tmp42 ? tmp67 : tmp68;
                                auto tmp70 = in_ptr8[static_cast<long>((-409600L) + x3 + (64L*x2) + (3200L*x1) + (25600L*x0))];
                                auto tmp71 = decltype(tmp69)(tmp69 + tmp70);
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp36 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp73 = tmp27 ? tmp35 : tmp72;
                            auto tmp74 = tmp4 ? tmp23 : tmp73;
                            out_ptr1[static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0))] = tmp74;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(50L))) + (3200L*(static_cast<long>(c10::div_floor_integer((x1 + x1_inner), 64L)) % static_cast<long>(8L))) + (25600L*(c10::div_floor_integer(x0, 50L))) + (204800L*(c10::div_floor_integer((x1 + x1_inner), 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_select_backward_slice_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr0 + static_cast<long>(512L + x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25600L*x0)), to_float_mask(tmp2));
                            auto tmp5 = masked_load(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0)), to_float_mask(tmp2));
                            auto tmp6 = tmp4 + tmp5;
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = to_float_mask(tmp2);
                        auto tmp10 = at::vec::Vectorized<float>(tmp8);
                        auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = masked_load(in_ptr0 + static_cast<long>(x2 + (25600L*x0)), to_float_mask(tmp12));
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                        auto tmp16 = to_float_mask(tmp12);
                        auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                        auto tmp18 = tmp11 + tmp17;
                        tmp18.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(819200L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_11 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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


cpp_fused_convolution_backward_sum_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(320L + x2 + (1536L*x1) + (76800L*x0)));
                            auto tmp0 = c10::convert<int>(1L + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = masked_load(in_ptr1 + static_cast<long>(832L + x2 + (512L*x1) + (25600L*x0)), to_float_mask(tmp2));
                                return tmp4;
                            }
                            ;
                            auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                            auto tmp7 = tmp5 * tmp6;
                            tmp7.store(out_ptr1 + static_cast<long>(x2 + (192L*x1) + (9408L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (1536L*x1) + (76800L*x0)));
                    auto tmp0 = c10::convert<int>(1L + x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = masked_load(in_ptr2 + static_cast<long>(640L + x2 + (512L*x1) + (25600L*x0)), to_float_mask(tmp2));
                        return tmp4;
                    }
                    ;
                    auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp7 = tmp5 * tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x2 + (192L*x1) + (9408L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4800L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                {
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x1) + (76800L*x0)));
                    auto tmp0 = c10::convert<int>(1L + x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = masked_load(in_ptr2 + static_cast<long>(512L + x2 + (512L*x1) + (25600L*x0)), to_float_mask(tmp2));
                        return tmp4;
                    }
                    ;
                    auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp7 = tmp5 * tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (6272L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_mul_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25600L*x0)));
                            auto tmp1 = static_cast<float>(0.125);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_stack_view_16 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (50L*x2) + (3200L*x1) + (25600L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x2) + (512L*x3) + (512L*x3_inner) + (25600L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(50L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (50L*x2) + (3200L*x1) + (25600L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x2) + (512L*x3) + (25600L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (64L*x1) + (512L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x0);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x2);
                                auto tmp7 = static_cast<long>(1);
                                auto tmp8 = tmp6 >= tmp7;
                                auto tmp9 = [&]
                                {
                                    auto tmp10 = c10::convert<long>(x2);
                                    auto tmp11 = static_cast<long>(0);
                                    auto tmp12 = tmp10 >= tmp11;
                                    auto tmp13 = [&]
                                    {
                                        auto tmp14 = in_ptr2[static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25600L*x0))];
                                        return tmp14;
                                    }
                                    ;
                                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                    auto tmp16 = in_ptr3[static_cast<long>(x3 + (64L*x1) + (512L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (25088L*x0))];
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                                auto tmp19 = static_cast<float>(0.0);
                                auto tmp20 = tmp8 ? tmp18 : tmp19;
                                auto tmp21 = in_ptr4[static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0))];
                                auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp24 = tmp0 >= tmp3;
                            auto tmp25 = static_cast<long>(16);
                            auto tmp26 = tmp0 < tmp25;
                            auto tmp27 = tmp24 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr0[static_cast<long>((-204800L) + x2 + (50L*x3) + (3200L*x1) + (25600L*x0))];
                                auto tmp30 = in_ptr1[static_cast<long>((-204800L) + x1 + (8L*x3) + (512L*x2) + (25600L*x0))];
                                auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                                auto tmp32 = out_ptr0[static_cast<long>((-4096L) + x3 + (64L*x1) + (512L*x0))];
                                auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                                auto tmp34 = decltype(tmp31)(tmp31 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp36 = tmp0 >= tmp25;
                            auto tmp37 = static_cast<long>(24);
                            auto tmp38 = tmp0 < tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x2);
                                auto tmp41 = static_cast<long>(1);
                                auto tmp42 = tmp40 >= tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = c10::convert<long>(x3 + (64L*x1));
                                    auto tmp45 = static_cast<long>(0);
                                    auto tmp46 = tmp44 >= tmp45;
                                    auto tmp47 = static_cast<long>(128);
                                    auto tmp48 = tmp44 < tmp47;
                                    auto tmp49 = [&]
                                    {
                                        auto tmp50 = in_ptr5[static_cast<long>((-100352L) + x3 + (64L*x1) + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (6272L*x0))];
                                        return tmp50;
                                    }
                                    ;
                                    auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                    auto tmp52 = tmp44 >= tmp47;
                                    auto tmp53 = static_cast<long>(320);
                                    auto tmp54 = tmp44 < tmp53;
                                    auto tmp55 = tmp52 & tmp54;
                                    auto tmp56 = [&]
                                    {
                                        auto tmp57 = in_ptr6[static_cast<long>((-150656L) + x3 + (64L*x1) + (192L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (9408L*x0))];
                                        return tmp57;
                                    }
                                    ;
                                    auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                                    auto tmp59 = tmp44 >= tmp53;
                                    auto tmp60 = static_cast<long>(512);
                                    auto tmp61 = tmp44 < tmp60;
                                    auto tmp62 = [&]
                                    {
                                        auto tmp63 = in_ptr7[static_cast<long>((-150848L) + x3 + (64L*x1) + (192L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (9408L*x0))];
                                        return tmp63;
                                    }
                                    ;
                                    auto tmp64 = tmp59 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                                    auto tmp65 = tmp55 ? tmp58 : tmp64;
                                    auto tmp66 = tmp48 ? tmp51 : tmp65;
                                    return tmp66;
                                }
                                ;
                                auto tmp67 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp68 = static_cast<float>(0.0);
                                auto tmp69 = tmp42 ? tmp67 : tmp68;
                                auto tmp70 = in_ptr8[static_cast<long>((-409600L) + x3 + (64L*x2) + (3200L*x1) + (25600L*x0))];
                                auto tmp71 = decltype(tmp69)(tmp69 + tmp70);
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp36 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp73 = tmp27 ? tmp35 : tmp72;
                            auto tmp74 = tmp4 ? tmp23 : tmp73;
                            out_ptr1[static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0))] = tmp74;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(50L))) + (3200L*(static_cast<long>(c10::div_floor_integer((x1 + x1_inner), 64L)) % static_cast<long>(8L))) + (25600L*(c10::div_floor_integer(x0, 50L))) + (204800L*(c10::div_floor_integer((x1 + x1_inner), 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
    }
}
''')


cpp_fused_add_convolution_backward_native_layer_norm_backward_sum_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr3 = in_out_ptr2;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = static_cast<int>(0);
                    auto tmp1 = static_cast<int>(1);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = masked_load(in_ptr2 + static_cast<long>(25088L + x0 + (25600L*x1)), to_float_mask(tmp2));
                        auto tmp5 = masked_load(in_ptr3 + static_cast<long>(24576L + x0 + (25088L*x1)), to_float_mask(tmp2));
                        auto tmp6 = tmp4 + tmp5;
                        return tmp6;
                    }
                    ;
                    auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp9 = to_float_mask(tmp2);
                    auto tmp10 = at::vec::Vectorized<float>(tmp8);
                    auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                    auto tmp12 = tmp0 < tmp1;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x0 + (25600L*x1)), to_float_mask(tmp12));
                        return tmp14;
                    }
                    ;
                    auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                    auto tmp16 = to_float_mask(tmp12);
                    auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                    auto tmp18 = tmp11 + tmp17;
                    tmp_acc0_vec = tmp_acc0_vec + tmp18;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (512L*x1) + (25088L*x0)));
                            auto tmp0 = c10::convert<int>(1L + x1);
                            auto tmp1 = static_cast<int>(1);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = masked_load(in_ptr2 + static_cast<long>(512L + x2 + (512L*x1) + (25600L*x0)), to_float_mask(tmp2));
                                auto tmp5 = masked_load(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (25088L*x0)), to_float_mask(tmp2));
                                auto tmp6 = tmp4 + tmp5;
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                            auto tmp8 = static_cast<float>(0.0);
                            auto tmp9 = to_float_mask(tmp2);
                            auto tmp10 = at::vec::Vectorized<float>(tmp8);
                            auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                            auto tmp12 = tmp0 < tmp1;
                            auto tmp13 = [&]
                            {
                                auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x2 + (25600L*x0)), to_float_mask(tmp12));
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                            auto tmp16 = to_float_mask(tmp12);
                            auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                            auto tmp18 = tmp11 + tmp17;
                            auto tmp20 = tmp18 * tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr2 + static_cast<long>(512L + x2 + (512L*x1) + (25600L*x0)), to_float_mask(tmp2));
                                auto tmp23 = masked_load(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (25088L*x0)), to_float_mask(tmp2));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp2));
                            auto tmp26 = decltype(tmp25)::blendv(tmp10, tmp25, tmp9);
                            auto tmp27 = [&]
                            {
                                auto tmp28 = masked_load(in_ptr2 + static_cast<long>(x2 + (25600L*x0)), to_float_mask(tmp12));
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp27())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp27(), to_float_mask(tmp12));
                            auto tmp30 = decltype(tmp29)::blendv(tmp10, tmp29, tmp16);
                            auto tmp31 = tmp26 + tmp30;
                            auto tmp32 = tmp31 * tmp19;
                            auto tmp34 = tmp32 * tmp33;
                            tmp_acc0_vec = tmp_acc0_vec + tmp20;
                            tmp_acc1_vec = tmp_acc1_vec + tmp34;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr1[static_cast<long>(x1 + (49L*x0))] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr2[static_cast<long>(x1 + (49L*x0))] = static_cast<float>(tmp_acc1);
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x2) + (25088L*x1)));
                            auto tmp0 = c10::convert<int>(1L + x2);
                            auto tmp1 = static_cast<int>(1);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = masked_load(in_ptr2 + static_cast<long>(512L + x0 + (512L*x2) + (25600L*x1)), to_float_mask(tmp2));
                                auto tmp5 = masked_load(in_ptr3 + static_cast<long>(x0 + (512L*x2) + (25088L*x1)), to_float_mask(tmp2));
                                auto tmp6 = tmp4 + tmp5;
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                            auto tmp8 = static_cast<float>(0.0);
                            auto tmp9 = to_float_mask(tmp2);
                            auto tmp10 = at::vec::Vectorized<float>(tmp8);
                            auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                            auto tmp12 = tmp0 < tmp1;
                            auto tmp13 = [&]
                            {
                                auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x0 + (25600L*x1)), to_float_mask(tmp12));
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                            auto tmp16 = to_float_mask(tmp12);
                            auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                            auto tmp18 = tmp11 + tmp17;
                            auto tmp20 = tmp18 * tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr2 + static_cast<long>(512L + x0 + (512L*x2) + (25600L*x1)), to_float_mask(tmp2));
                                auto tmp23 = masked_load(in_ptr3 + static_cast<long>(x0 + (512L*x2) + (25088L*x1)), to_float_mask(tmp2));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp2));
                            auto tmp26 = decltype(tmp25)::blendv(tmp10, tmp25, tmp9);
                            auto tmp27 = [&]
                            {
                                auto tmp28 = masked_load(in_ptr2 + static_cast<long>(x0 + (25600L*x1)), to_float_mask(tmp12));
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp27())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp27(), to_float_mask(tmp12));
                            auto tmp30 = decltype(tmp29)::blendv(tmp10, tmp29, tmp16);
                            auto tmp31 = tmp26 + tmp30;
                            tmp_acc0_vec = tmp_acc0_vec + tmp20;
                            tmp_acc1_vec = tmp_acc1_vec + tmp31;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr6[static_cast<long>(x1 + (49L*x0))];
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp25 = out_ptr1[static_cast<long>(x1 + (49L*x0))];
                        auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (512L*x1) + (25088L*x0)));
                        auto tmp29 = out_ptr2[static_cast<long>(x1 + (49L*x0))];
                        auto tmp1 = c10::convert<int>(1L + x1);
                        auto tmp2 = static_cast<int>(1);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = masked_load(in_ptr2 + static_cast<long>(512L + x2 + (512L*x1) + (25600L*x0)), to_float_mask(tmp3));
                            auto tmp6 = masked_load(in_out_ptr2 + static_cast<long>(x2 + (512L*x1) + (25088L*x0)), to_float_mask(tmp3));
                            auto tmp7 = tmp5 + tmp6;
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp4())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp4(), to_float_mask(tmp3));
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = to_float_mask(tmp3);
                        auto tmp11 = at::vec::Vectorized<float>(tmp9);
                        auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                        auto tmp13 = tmp1 < tmp2;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = masked_load(in_ptr2 + static_cast<long>(x2 + (25600L*x0)), to_float_mask(tmp13));
                            return tmp15;
                        }
                        ;
                        auto tmp16 = decltype(tmp14())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp14(), to_float_mask(tmp13));
                        auto tmp17 = to_float_mask(tmp13);
                        auto tmp18 = decltype(tmp16)::blendv(tmp11, tmp16, tmp17);
                        auto tmp19 = tmp12 + tmp18;
                        auto tmp21 = tmp19 * tmp20;
                        auto tmp22 = static_cast<float>(512.0);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = at::vec::Vectorized<float>(tmp25);
                        auto tmp27 = tmp24 - tmp26;
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        auto tmp32 = tmp27 - tmp31;
                        auto tmp33 = at::vec::Vectorized<float>(tmp0);
                        auto tmp34 = tmp33 * tmp32;
                        tmp34.store(in_out_ptr2 + static_cast<long>(x2 + (512L*x1) + (25088L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_slice_backward_view_19 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(static_cast<long>(x0) % static_cast<long>(197L));
                    auto tmp1 = static_cast<int>(1);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = masked_load(in_ptr0 + static_cast<long>((-320L) + x1 + (320L*(static_cast<long>(x0) % static_cast<long>(197L))) + (62720L*(c10::div_floor_integer(x0, 197L)))), to_float_mask(tmp2));
                        return tmp4;
                    }
                    ;
                    auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = to_float_mask(tmp2);
                    auto tmp8 = at::vec::Vectorized<float>(tmp6);
                    auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp7);
                    tmp9.store(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (320L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2017280L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_slice_backward_sum_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1280L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (320L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (320L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        auto tmp10 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp17 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                        auto tmp21 = out_ptr2[static_cast<long>(x1 + (197L*x0))];
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr4 + static_cast<long>((-320L) + x2 + (320L*x1) + (62720L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = to_float_mask(tmp2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp6);
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp7);
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp14 = static_cast<float>(320.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp22 = at::vec::Vectorized<float>(tmp21);
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp19 - tmp23;
                        auto tmp25 = at::vec::Vectorized<float>(tmp10);
                        auto tmp26 = tmp25 * tmp24;
                        auto tmp27 = tmp9 + tmp26;
                        tmp27.store(in_out_ptr0 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_sum_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (320L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(200L + x2 + (960L*x1) + (189120L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr1 + static_cast<long>(520L + x2 + (320L*x1) + (63040L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr1 + static_cast<long>(x2 + (120L*x1) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(80L + x2 + (960L*x1) + (189120L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr0 + static_cast<long>(400L + x2 + (320L*x1) + (63040L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (120L*x1) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (960L*x1) + (189120L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr0 + static_cast<long>(320L + x2 + (320L*x1) + (63040L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (15680L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (40L*x1) + (320L*x2) + (63040L*x0)));
                            auto tmp1 = static_cast<float>(0.15811388300841897);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_stack_view_26 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(40L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (7880L*x1) + (63040L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x2) + (320L*x3) + (320L*x3_inner) + (63040L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (7880L*x1) + (63040L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x2) + (320L*x3) + (63040L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (40L*x1) + (320L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x0);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x2);
                                auto tmp7 = static_cast<long>(1);
                                auto tmp8 = tmp6 >= tmp7;
                                auto tmp9 = [&]
                                {
                                    auto tmp10 = c10::convert<long>(x2);
                                    auto tmp11 = static_cast<long>(0);
                                    auto tmp12 = tmp10 >= tmp11;
                                    auto tmp13 = [&]
                                    {
                                        auto tmp14 = in_ptr2[static_cast<long>(x3 + (40L*x1) + (320L*x2) + (63040L*x0))];
                                        return tmp14;
                                    }
                                    ;
                                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                    auto tmp16 = in_ptr3[static_cast<long>(x3 + (40L*x1) + (320L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (62720L*x0))];
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                                auto tmp19 = static_cast<float>(0.0);
                                auto tmp20 = tmp8 ? tmp18 : tmp19;
                                auto tmp21 = in_ptr4[static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0))];
                                auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp24 = tmp0 >= tmp3;
                            auto tmp25 = static_cast<long>(16);
                            auto tmp26 = tmp0 < tmp25;
                            auto tmp27 = tmp24 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr0[static_cast<long>((-504320L) + x2 + (197L*x3) + (7880L*x1) + (63040L*x0))];
                                auto tmp30 = in_ptr1[static_cast<long>((-504320L) + x1 + (8L*x3) + (320L*x2) + (63040L*x0))];
                                auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                                auto tmp32 = out_ptr0[static_cast<long>((-2560L) + x3 + (40L*x1) + (320L*x0))];
                                auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                                auto tmp34 = decltype(tmp31)(tmp31 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp36 = tmp0 >= tmp25;
                            auto tmp37 = static_cast<long>(24);
                            auto tmp38 = tmp0 < tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x2);
                                auto tmp41 = static_cast<long>(1);
                                auto tmp42 = tmp40 >= tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = c10::convert<long>(x3 + (40L*x1));
                                    auto tmp45 = static_cast<long>(0);
                                    auto tmp46 = tmp44 >= tmp45;
                                    auto tmp47 = static_cast<long>(80);
                                    auto tmp48 = tmp44 < tmp47;
                                    auto tmp49 = [&]
                                    {
                                        auto tmp50 = in_ptr5[static_cast<long>((-250880L) + x3 + (40L*x1) + (80L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (15680L*x0))];
                                        return tmp50;
                                    }
                                    ;
                                    auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                    auto tmp52 = tmp44 >= tmp47;
                                    auto tmp53 = static_cast<long>(200);
                                    auto tmp54 = tmp44 < tmp53;
                                    auto tmp55 = tmp52 & tmp54;
                                    auto tmp56 = [&]
                                    {
                                        auto tmp57 = in_ptr6[static_cast<long>((-376400L) + x3 + (40L*x1) + (120L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (23520L*x0))];
                                        return tmp57;
                                    }
                                    ;
                                    auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                                    auto tmp59 = tmp44 >= tmp53;
                                    auto tmp60 = static_cast<long>(320);
                                    auto tmp61 = tmp44 < tmp60;
                                    auto tmp62 = [&]
                                    {
                                        auto tmp63 = in_ptr7[static_cast<long>((-376520L) + x3 + (40L*x1) + (120L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (23520L*x0))];
                                        return tmp63;
                                    }
                                    ;
                                    auto tmp64 = tmp59 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                                    auto tmp65 = tmp55 ? tmp58 : tmp64;
                                    auto tmp66 = tmp48 ? tmp51 : tmp65;
                                    return tmp66;
                                }
                                ;
                                auto tmp67 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp68 = static_cast<float>(0.0);
                                auto tmp69 = tmp42 ? tmp67 : tmp68;
                                auto tmp70 = in_ptr8[static_cast<long>((-1008640L) + x3 + (40L*x2) + (7880L*x1) + (63040L*x0))];
                                auto tmp71 = decltype(tmp69)(tmp69 + tmp70);
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp36 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp73 = tmp27 ? tmp35 : tmp72;
                            auto tmp74 = tmp4 ? tmp23 : tmp73;
                            out_ptr1[static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0))] = tmp74;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>((40L*(static_cast<long>(x0) % static_cast<long>(197L))) + (7880L*(static_cast<long>(c10::div_floor_integer((x1 + x1_inner), 40L)) % static_cast<long>(8L))) + (63040L*(c10::div_floor_integer(x0, 197L))) + (504320L*(c10::div_floor_integer((x1 + x1_inner), 320L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(40L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (320L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (320L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp13 = in_ptr4[static_cast<long>(x0)];
                    auto tmp18 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(320.0);
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
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_slice_backward_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr0 + static_cast<long>(320L + x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (63040L*x0)), to_float_mask(tmp2));
                            auto tmp5 = masked_load(in_ptr1 + static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0)), to_float_mask(tmp2));
                            auto tmp6 = tmp4 + tmp5;
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = to_float_mask(tmp2);
                        auto tmp10 = at::vec::Vectorized<float>(tmp8);
                        auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = masked_load(in_ptr0 + static_cast<long>(x2 + (63040L*x0)), to_float_mask(tmp12));
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                        auto tmp16 = to_float_mask(tmp12);
                        auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                        auto tmp18 = tmp11 + tmp17;
                        tmp18.store(out_ptr0 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (320L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2017280L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1280L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (320L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (320L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(320.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_sum_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (320L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(200L + x2 + (960L*x1) + (189120L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr1 + static_cast<long>(520L + x2 + (320L*x1) + (63040L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr1 + static_cast<long>(x2 + (120L*x1) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(5880L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(80L + x2 + (960L*x1) + (189120L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr2 + static_cast<long>(400L + x2 + (320L*x1) + (63040L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (120L*x1) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3000L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (960L*x1) + (189120L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr2 + static_cast<long>(320L + x2 + (320L*x1) + (63040L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (15680L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_mul_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (40L*x1) + (320L*x2) + (63040L*x0)));
                            auto tmp1 = static_cast<float>(0.15811388300841897);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_stack_view_35 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(40L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (7880L*x1) + (63040L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x2) + (320L*x3) + (320L*x3_inner) + (63040L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (7880L*x1) + (63040L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x2) + (320L*x3) + (63040L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (40L*x1) + (320L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x0);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x2);
                                auto tmp7 = static_cast<long>(1);
                                auto tmp8 = tmp6 >= tmp7;
                                auto tmp9 = [&]
                                {
                                    auto tmp10 = c10::convert<long>(x2);
                                    auto tmp11 = static_cast<long>(0);
                                    auto tmp12 = tmp10 >= tmp11;
                                    auto tmp13 = [&]
                                    {
                                        auto tmp14 = in_ptr2[static_cast<long>(x3 + (40L*x1) + (320L*x2) + (63040L*x0))];
                                        return tmp14;
                                    }
                                    ;
                                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                    auto tmp16 = in_ptr3[static_cast<long>(x3 + (40L*x1) + (320L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (62720L*x0))];
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                                auto tmp19 = static_cast<float>(0.0);
                                auto tmp20 = tmp8 ? tmp18 : tmp19;
                                auto tmp21 = in_ptr4[static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0))];
                                auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp24 = tmp0 >= tmp3;
                            auto tmp25 = static_cast<long>(16);
                            auto tmp26 = tmp0 < tmp25;
                            auto tmp27 = tmp24 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr0[static_cast<long>((-504320L) + x2 + (197L*x3) + (7880L*x1) + (63040L*x0))];
                                auto tmp30 = in_ptr1[static_cast<long>((-504320L) + x1 + (8L*x3) + (320L*x2) + (63040L*x0))];
                                auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                                auto tmp32 = out_ptr0[static_cast<long>((-2560L) + x3 + (40L*x1) + (320L*x0))];
                                auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                                auto tmp34 = decltype(tmp31)(tmp31 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp36 = tmp0 >= tmp25;
                            auto tmp37 = static_cast<long>(24);
                            auto tmp38 = tmp0 < tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x2);
                                auto tmp41 = static_cast<long>(1);
                                auto tmp42 = tmp40 >= tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = c10::convert<long>(x3 + (40L*x1));
                                    auto tmp45 = static_cast<long>(0);
                                    auto tmp46 = tmp44 >= tmp45;
                                    auto tmp47 = static_cast<long>(80);
                                    auto tmp48 = tmp44 < tmp47;
                                    auto tmp49 = [&]
                                    {
                                        auto tmp50 = in_ptr5[static_cast<long>((-250880L) + x3 + (40L*x1) + (80L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (15680L*x0))];
                                        return tmp50;
                                    }
                                    ;
                                    auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                    auto tmp52 = tmp44 >= tmp47;
                                    auto tmp53 = static_cast<long>(200);
                                    auto tmp54 = tmp44 < tmp53;
                                    auto tmp55 = tmp52 & tmp54;
                                    auto tmp56 = [&]
                                    {
                                        auto tmp57 = in_ptr6[static_cast<long>((-376400L) + x3 + (40L*x1) + (120L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (23520L*x0))];
                                        return tmp57;
                                    }
                                    ;
                                    auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                                    auto tmp59 = tmp44 >= tmp53;
                                    auto tmp60 = static_cast<long>(320);
                                    auto tmp61 = tmp44 < tmp60;
                                    auto tmp62 = [&]
                                    {
                                        auto tmp63 = in_ptr7[static_cast<long>((-376520L) + x3 + (40L*x1) + (120L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (23520L*x0))];
                                        return tmp63;
                                    }
                                    ;
                                    auto tmp64 = tmp59 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                                    auto tmp65 = tmp55 ? tmp58 : tmp64;
                                    auto tmp66 = tmp48 ? tmp51 : tmp65;
                                    return tmp66;
                                }
                                ;
                                auto tmp67 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp68 = static_cast<float>(0.0);
                                auto tmp69 = tmp42 ? tmp67 : tmp68;
                                auto tmp70 = in_ptr8[static_cast<long>((-1008640L) + x3 + (40L*x2) + (7880L*x1) + (63040L*x0))];
                                auto tmp71 = decltype(tmp69)(tmp69 + tmp70);
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp36 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp73 = tmp27 ? tmp35 : tmp72;
                            auto tmp74 = tmp4 ? tmp23 : tmp73;
                            out_ptr1[static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0))] = tmp74;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>((40L*(static_cast<long>(x0) % static_cast<long>(197L))) + (7880L*(static_cast<long>(c10::div_floor_integer((x1 + x1_inner), 40L)) % static_cast<long>(8L))) + (63040L*(c10::div_floor_integer(x0, 197L))) + (504320L*(c10::div_floor_integer((x1 + x1_inner), 320L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(40L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (320L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (320L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp13 = in_ptr4[static_cast<long>(x0)];
                    auto tmp18 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(320.0);
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
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_layer_norm_backward_sum_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr3 = in_out_ptr2;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = static_cast<int>(0);
                    auto tmp1 = static_cast<int>(1);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = masked_load(in_ptr2 + static_cast<long>(62720L + x0 + (63040L*x1)), to_float_mask(tmp2));
                        auto tmp5 = masked_load(in_ptr3 + static_cast<long>(62400L + x0 + (62720L*x1)), to_float_mask(tmp2));
                        auto tmp6 = tmp4 + tmp5;
                        return tmp6;
                    }
                    ;
                    auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp9 = to_float_mask(tmp2);
                    auto tmp10 = at::vec::Vectorized<float>(tmp8);
                    auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                    auto tmp12 = tmp0 < tmp1;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x0 + (63040L*x1)), to_float_mask(tmp12));
                        return tmp14;
                    }
                    ;
                    auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                    auto tmp16 = to_float_mask(tmp12);
                    auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                    auto tmp18 = tmp11 + tmp17;
                    tmp_acc0_vec = tmp_acc0_vec + tmp18;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                        {
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (320L*x1) + (62720L*x0)));
                            auto tmp0 = c10::convert<int>(1L + x1);
                            auto tmp1 = static_cast<int>(1);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = masked_load(in_ptr2 + static_cast<long>(320L + x2 + (320L*x1) + (63040L*x0)), to_float_mask(tmp2));
                                auto tmp5 = masked_load(in_ptr3 + static_cast<long>(x2 + (320L*x1) + (62720L*x0)), to_float_mask(tmp2));
                                auto tmp6 = tmp4 + tmp5;
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                            auto tmp8 = static_cast<float>(0.0);
                            auto tmp9 = to_float_mask(tmp2);
                            auto tmp10 = at::vec::Vectorized<float>(tmp8);
                            auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                            auto tmp12 = tmp0 < tmp1;
                            auto tmp13 = [&]
                            {
                                auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x2 + (63040L*x0)), to_float_mask(tmp12));
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                            auto tmp16 = to_float_mask(tmp12);
                            auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                            auto tmp18 = tmp11 + tmp17;
                            auto tmp20 = tmp18 * tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr2 + static_cast<long>(320L + x2 + (320L*x1) + (63040L*x0)), to_float_mask(tmp2));
                                auto tmp23 = masked_load(in_ptr3 + static_cast<long>(x2 + (320L*x1) + (62720L*x0)), to_float_mask(tmp2));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp2));
                            auto tmp26 = decltype(tmp25)::blendv(tmp10, tmp25, tmp9);
                            auto tmp27 = [&]
                            {
                                auto tmp28 = masked_load(in_ptr2 + static_cast<long>(x2 + (63040L*x0)), to_float_mask(tmp12));
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp27())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp27(), to_float_mask(tmp12));
                            auto tmp30 = decltype(tmp29)::blendv(tmp10, tmp29, tmp16);
                            auto tmp31 = tmp26 + tmp30;
                            auto tmp32 = tmp31 * tmp19;
                            auto tmp34 = tmp32 * tmp33;
                            tmp_acc0_vec = tmp_acc0_vec + tmp20;
                            tmp_acc1_vec = tmp_acc1_vec + tmp34;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr2[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc1);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (320L*x2) + (62720L*x1)));
                            auto tmp0 = c10::convert<int>(1L + x2);
                            auto tmp1 = static_cast<int>(1);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = masked_load(in_ptr2 + static_cast<long>(320L + x0 + (320L*x2) + (63040L*x1)), to_float_mask(tmp2));
                                auto tmp5 = masked_load(in_ptr3 + static_cast<long>(x0 + (320L*x2) + (62720L*x1)), to_float_mask(tmp2));
                                auto tmp6 = tmp4 + tmp5;
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                            auto tmp8 = static_cast<float>(0.0);
                            auto tmp9 = to_float_mask(tmp2);
                            auto tmp10 = at::vec::Vectorized<float>(tmp8);
                            auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                            auto tmp12 = tmp0 < tmp1;
                            auto tmp13 = [&]
                            {
                                auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x0 + (63040L*x1)), to_float_mask(tmp12));
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                            auto tmp16 = to_float_mask(tmp12);
                            auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                            auto tmp18 = tmp11 + tmp17;
                            auto tmp20 = tmp18 * tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr2 + static_cast<long>(320L + x0 + (320L*x2) + (63040L*x1)), to_float_mask(tmp2));
                                auto tmp23 = masked_load(in_ptr3 + static_cast<long>(x0 + (320L*x2) + (62720L*x1)), to_float_mask(tmp2));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp2));
                            auto tmp26 = decltype(tmp25)::blendv(tmp10, tmp25, tmp9);
                            auto tmp27 = [&]
                            {
                                auto tmp28 = masked_load(in_ptr2 + static_cast<long>(x0 + (63040L*x1)), to_float_mask(tmp12));
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp27())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp27(), to_float_mask(tmp12));
                            auto tmp30 = decltype(tmp29)::blendv(tmp10, tmp29, tmp16);
                            auto tmp31 = tmp26 + tmp30;
                            tmp_acc0_vec = tmp_acc0_vec + tmp20;
                            tmp_acc1_vec = tmp_acc1_vec + tmp31;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (320L*x1) + (62720L*x0)));
                        auto tmp29 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = c10::convert<int>(1L + x1);
                        auto tmp2 = static_cast<int>(1);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = masked_load(in_ptr2 + static_cast<long>(320L + x2 + (320L*x1) + (63040L*x0)), to_float_mask(tmp3));
                            auto tmp6 = masked_load(in_out_ptr2 + static_cast<long>(x2 + (320L*x1) + (62720L*x0)), to_float_mask(tmp3));
                            auto tmp7 = tmp5 + tmp6;
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp4())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp4(), to_float_mask(tmp3));
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = to_float_mask(tmp3);
                        auto tmp11 = at::vec::Vectorized<float>(tmp9);
                        auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                        auto tmp13 = tmp1 < tmp2;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = masked_load(in_ptr2 + static_cast<long>(x2 + (63040L*x0)), to_float_mask(tmp13));
                            return tmp15;
                        }
                        ;
                        auto tmp16 = decltype(tmp14())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp14(), to_float_mask(tmp13));
                        auto tmp17 = to_float_mask(tmp13);
                        auto tmp18 = decltype(tmp16)::blendv(tmp11, tmp16, tmp17);
                        auto tmp19 = tmp12 + tmp18;
                        auto tmp21 = tmp19 * tmp20;
                        auto tmp22 = static_cast<float>(320.0);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = at::vec::Vectorized<float>(tmp25);
                        auto tmp27 = tmp24 - tmp26;
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        auto tmp32 = tmp27 - tmp31;
                        auto tmp33 = at::vec::Vectorized<float>(tmp0);
                        auto tmp34 = tmp33 * tmp32;
                        tmp34.store(in_out_ptr2 + static_cast<long>(x2 + (320L*x1) + (62720L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_slice_backward_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(static_cast<long>(x0) % static_cast<long>(785L));
                    auto tmp1 = static_cast<int>(1);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = masked_load(in_ptr0 + static_cast<long>((-128L) + x1 + (128L*(static_cast<long>(x0) % static_cast<long>(785L))) + (100352L*(c10::div_floor_integer(x0, 785L)))), to_float_mask(tmp2));
                        return tmp4;
                    }
                    ;
                    auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = to_float_mask(tmp2);
                    auto tmp8 = at::vec::Vectorized<float>(tmp6);
                    auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp7);
                    tmp9.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_39 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6280L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6430720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_slice_backward_sum_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6280L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6280L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp10 = in_ptr5[static_cast<long>(x1 + (785L*x0))];
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp17 = out_ptr1[static_cast<long>(x1 + (785L*x0))];
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                        auto tmp21 = out_ptr2[static_cast<long>(x1 + (785L*x0))];
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr4 + static_cast<long>((-128L) + x2 + (128L*x1) + (100352L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = to_float_mask(tmp2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp6);
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp7);
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp14 = static_cast<float>(128.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp22 = at::vec::Vectorized<float>(tmp21);
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp19 - tmp23;
                        auto tmp25 = at::vec::Vectorized<float>(tmp10);
                        auto tmp26 = tmp25 * tmp24;
                        auto tmp27 = tmp9 + tmp26;
                        tmp27.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_sum_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6280L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(80L + x2 + (384L*x1) + (301440L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr1 + static_cast<long>(208L + x2 + (128L*x1) + (100480L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr1 + static_cast<long>(x2 + (48L*x1) + (37632L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(32L + x2 + (384L*x1) + (301440L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr0 + static_cast<long>(160L + x2 + (128L*x1) + (100480L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (48L*x1) + (37632L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (301440L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr0 + static_cast<long>(128L + x2 + (128L*x1) + (100480L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (25088L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x1) + (128L*x2) + (100480L*x0)));
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_stack_view_45 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(784L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (785L*x2) + (12560L*x1) + (100480L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x2) + (128L*x3) + (128L*x3_inner) + (100480L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(784L); x3<static_cast<long>(785L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (785L*x2) + (12560L*x1) + (100480L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x2) + (128L*x3) + (100480L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (16L*x1) + (128L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x0);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x2);
                                auto tmp7 = static_cast<long>(1);
                                auto tmp8 = tmp6 >= tmp7;
                                auto tmp9 = [&]
                                {
                                    auto tmp10 = c10::convert<long>(x2);
                                    auto tmp11 = static_cast<long>(0);
                                    auto tmp12 = tmp10 >= tmp11;
                                    auto tmp13 = [&]
                                    {
                                        auto tmp14 = in_ptr2[static_cast<long>(x3 + (16L*x1) + (128L*x2) + (100480L*x0))];
                                        return tmp14;
                                    }
                                    ;
                                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                    auto tmp16 = in_ptr3[static_cast<long>(x3 + (16L*x1) + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (100352L*x0))];
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                                auto tmp19 = static_cast<float>(0.0);
                                auto tmp20 = tmp8 ? tmp18 : tmp19;
                                auto tmp21 = in_ptr4[static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0))];
                                auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp24 = tmp0 >= tmp3;
                            auto tmp25 = static_cast<long>(16);
                            auto tmp26 = tmp0 < tmp25;
                            auto tmp27 = tmp24 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr0[static_cast<long>((-803840L) + x2 + (785L*x3) + (12560L*x1) + (100480L*x0))];
                                auto tmp30 = in_ptr1[static_cast<long>((-803840L) + x1 + (8L*x3) + (128L*x2) + (100480L*x0))];
                                auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                                auto tmp32 = out_ptr0[static_cast<long>((-1024L) + x3 + (16L*x1) + (128L*x0))];
                                auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                                auto tmp34 = decltype(tmp31)(tmp31 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp36 = tmp0 >= tmp25;
                            auto tmp37 = static_cast<long>(24);
                            auto tmp38 = tmp0 < tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x2);
                                auto tmp41 = static_cast<long>(1);
                                auto tmp42 = tmp40 >= tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = c10::convert<long>(x3 + (16L*x1));
                                    auto tmp45 = static_cast<long>(0);
                                    auto tmp46 = tmp44 >= tmp45;
                                    auto tmp47 = static_cast<long>(32);
                                    auto tmp48 = tmp44 < tmp47;
                                    auto tmp49 = [&]
                                    {
                                        auto tmp50 = in_ptr5[static_cast<long>((-401408L) + x3 + (16L*x1) + (32L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (25088L*x0))];
                                        return tmp50;
                                    }
                                    ;
                                    auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                    auto tmp52 = tmp44 >= tmp47;
                                    auto tmp53 = static_cast<long>(80);
                                    auto tmp54 = tmp44 < tmp53;
                                    auto tmp55 = tmp52 & tmp54;
                                    auto tmp56 = [&]
                                    {
                                        auto tmp57 = in_ptr6[static_cast<long>((-602144L) + x3 + (16L*x1) + (48L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (37632L*x0))];
                                        return tmp57;
                                    }
                                    ;
                                    auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                                    auto tmp59 = tmp44 >= tmp53;
                                    auto tmp60 = static_cast<long>(128);
                                    auto tmp61 = tmp44 < tmp60;
                                    auto tmp62 = [&]
                                    {
                                        auto tmp63 = in_ptr7[static_cast<long>((-602192L) + x3 + (16L*x1) + (48L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (37632L*x0))];
                                        return tmp63;
                                    }
                                    ;
                                    auto tmp64 = tmp59 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                                    auto tmp65 = tmp55 ? tmp58 : tmp64;
                                    auto tmp66 = tmp48 ? tmp51 : tmp65;
                                    return tmp66;
                                }
                                ;
                                auto tmp67 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp68 = static_cast<float>(0.0);
                                auto tmp69 = tmp42 ? tmp67 : tmp68;
                                auto tmp70 = in_ptr8[static_cast<long>((-1607680L) + x3 + (16L*x2) + (12560L*x1) + (100480L*x0))];
                                auto tmp71 = decltype(tmp69)(tmp69 + tmp70);
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp36 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp73 = tmp27 ? tmp35 : tmp72;
                            auto tmp74 = tmp4 ? tmp23 : tmp73;
                            out_ptr1[static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0))] = tmp74;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(785L))) + (12560L*(static_cast<long>(c10::div_floor_integer((x1 + x1_inner), 16L)) % static_cast<long>(8L))) + (100480L*(c10::div_floor_integer(x0, 785L))) + (803840L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_46 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6280L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6280L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
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


cpp_fused_add_slice_backward_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr0 + static_cast<long>(128L + x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100480L*x0)), to_float_mask(tmp2));
                            auto tmp5 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0)), to_float_mask(tmp2));
                            auto tmp6 = tmp4 + tmp5;
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = to_float_mask(tmp2);
                        auto tmp10 = at::vec::Vectorized<float>(tmp8);
                        auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = masked_load(in_ptr0 + static_cast<long>(x2 + (100480L*x0)), to_float_mask(tmp12));
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                        auto tmp16 = to_float_mask(tmp12);
                        auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                        auto tmp18 = tmp11 + tmp17;
                        tmp18.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_48 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6280L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6430720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_49 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6280L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6280L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
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


cpp_fused_convolution_backward_sum_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6280L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(80L + x2 + (384L*x1) + (301440L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr1 + static_cast<long>(208L + x2 + (128L*x1) + (100480L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr1 + static_cast<long>(x2 + (48L*x1) + (37632L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x2 + (384L*x1) + (301440L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr2 + static_cast<long>(160L + x2 + (128L*x1) + (100480L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (48L*x1) + (37632L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1200L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (301440L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr2 + static_cast<long>(128L + x2 + (128L*x1) + (100480L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (25088L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_mul_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (16L*x1) + (128L*x2) + (100480L*x0)));
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_stack_view_54 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(784L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (785L*x2) + (12560L*x1) + (100480L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x2) + (128L*x3) + (128L*x3_inner) + (100480L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(784L); x3<static_cast<long>(785L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (785L*x2) + (12560L*x1) + (100480L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x2) + (128L*x3) + (100480L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (16L*x1) + (128L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x0);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x2);
                                auto tmp7 = static_cast<long>(1);
                                auto tmp8 = tmp6 >= tmp7;
                                auto tmp9 = [&]
                                {
                                    auto tmp10 = c10::convert<long>(x2);
                                    auto tmp11 = static_cast<long>(0);
                                    auto tmp12 = tmp10 >= tmp11;
                                    auto tmp13 = [&]
                                    {
                                        auto tmp14 = in_ptr2[static_cast<long>(x3 + (16L*x1) + (128L*x2) + (100480L*x0))];
                                        return tmp14;
                                    }
                                    ;
                                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                    auto tmp16 = in_ptr3[static_cast<long>(x3 + (16L*x1) + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (100352L*x0))];
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                                auto tmp19 = static_cast<float>(0.0);
                                auto tmp20 = tmp8 ? tmp18 : tmp19;
                                auto tmp21 = in_ptr4[static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0))];
                                auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp24 = tmp0 >= tmp3;
                            auto tmp25 = static_cast<long>(16);
                            auto tmp26 = tmp0 < tmp25;
                            auto tmp27 = tmp24 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr0[static_cast<long>((-803840L) + x2 + (785L*x3) + (12560L*x1) + (100480L*x0))];
                                auto tmp30 = in_ptr1[static_cast<long>((-803840L) + x1 + (8L*x3) + (128L*x2) + (100480L*x0))];
                                auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                                auto tmp32 = out_ptr0[static_cast<long>((-1024L) + x3 + (16L*x1) + (128L*x0))];
                                auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                                auto tmp34 = decltype(tmp31)(tmp31 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp36 = tmp0 >= tmp25;
                            auto tmp37 = static_cast<long>(24);
                            auto tmp38 = tmp0 < tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x2);
                                auto tmp41 = static_cast<long>(1);
                                auto tmp42 = tmp40 >= tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = c10::convert<long>(x3 + (16L*x1));
                                    auto tmp45 = static_cast<long>(0);
                                    auto tmp46 = tmp44 >= tmp45;
                                    auto tmp47 = static_cast<long>(32);
                                    auto tmp48 = tmp44 < tmp47;
                                    auto tmp49 = [&]
                                    {
                                        auto tmp50 = in_ptr5[static_cast<long>((-401408L) + x3 + (16L*x1) + (32L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (25088L*x0))];
                                        return tmp50;
                                    }
                                    ;
                                    auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                    auto tmp52 = tmp44 >= tmp47;
                                    auto tmp53 = static_cast<long>(80);
                                    auto tmp54 = tmp44 < tmp53;
                                    auto tmp55 = tmp52 & tmp54;
                                    auto tmp56 = [&]
                                    {
                                        auto tmp57 = in_ptr6[static_cast<long>((-602144L) + x3 + (16L*x1) + (48L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (37632L*x0))];
                                        return tmp57;
                                    }
                                    ;
                                    auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                                    auto tmp59 = tmp44 >= tmp53;
                                    auto tmp60 = static_cast<long>(128);
                                    auto tmp61 = tmp44 < tmp60;
                                    auto tmp62 = [&]
                                    {
                                        auto tmp63 = in_ptr7[static_cast<long>((-602192L) + x3 + (16L*x1) + (48L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (37632L*x0))];
                                        return tmp63;
                                    }
                                    ;
                                    auto tmp64 = tmp59 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                                    auto tmp65 = tmp55 ? tmp58 : tmp64;
                                    auto tmp66 = tmp48 ? tmp51 : tmp65;
                                    return tmp66;
                                }
                                ;
                                auto tmp67 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp68 = static_cast<float>(0.0);
                                auto tmp69 = tmp42 ? tmp67 : tmp68;
                                auto tmp70 = in_ptr8[static_cast<long>((-1607680L) + x3 + (16L*x2) + (12560L*x1) + (100480L*x0))];
                                auto tmp71 = decltype(tmp69)(tmp69 + tmp70);
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp36 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp73 = tmp27 ? tmp35 : tmp72;
                            auto tmp74 = tmp4 ? tmp23 : tmp73;
                            out_ptr1[static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0))] = tmp74;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(785L))) + (12560L*(static_cast<long>(c10::div_floor_integer((x1 + x1_inner), 16L)) % static_cast<long>(8L))) + (100480L*(c10::div_floor_integer(x0, 785L))) + (803840L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_55 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6280L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6280L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_layer_norm_backward_sum_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr3 = in_out_ptr2;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = static_cast<int>(0);
                    auto tmp1 = static_cast<int>(1);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = masked_load(in_ptr2 + static_cast<long>(100352L + x0 + (100480L*x1)), to_float_mask(tmp2));
                        auto tmp5 = masked_load(in_ptr3 + static_cast<long>(100224L + x0 + (100352L*x1)), to_float_mask(tmp2));
                        auto tmp6 = tmp4 + tmp5;
                        return tmp6;
                    }
                    ;
                    auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp9 = to_float_mask(tmp2);
                    auto tmp10 = at::vec::Vectorized<float>(tmp8);
                    auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                    auto tmp12 = tmp0 < tmp1;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x0 + (100480L*x1)), to_float_mask(tmp12));
                        return tmp14;
                    }
                    ;
                    auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                    auto tmp16 = to_float_mask(tmp12);
                    auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                    auto tmp18 = tmp11 + tmp17;
                    tmp_acc0_vec = tmp_acc0_vec + tmp18;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (128L*x1) + (100352L*x0)));
                            auto tmp0 = c10::convert<int>(1L + x1);
                            auto tmp1 = static_cast<int>(1);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = masked_load(in_ptr2 + static_cast<long>(128L + x2 + (128L*x1) + (100480L*x0)), to_float_mask(tmp2));
                                auto tmp5 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (100352L*x0)), to_float_mask(tmp2));
                                auto tmp6 = tmp4 + tmp5;
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                            auto tmp8 = static_cast<float>(0.0);
                            auto tmp9 = to_float_mask(tmp2);
                            auto tmp10 = at::vec::Vectorized<float>(tmp8);
                            auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                            auto tmp12 = tmp0 < tmp1;
                            auto tmp13 = [&]
                            {
                                auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x2 + (100480L*x0)), to_float_mask(tmp12));
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                            auto tmp16 = to_float_mask(tmp12);
                            auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                            auto tmp18 = tmp11 + tmp17;
                            auto tmp20 = tmp18 * tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr2 + static_cast<long>(128L + x2 + (128L*x1) + (100480L*x0)), to_float_mask(tmp2));
                                auto tmp23 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (100352L*x0)), to_float_mask(tmp2));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp2));
                            auto tmp26 = decltype(tmp25)::blendv(tmp10, tmp25, tmp9);
                            auto tmp27 = [&]
                            {
                                auto tmp28 = masked_load(in_ptr2 + static_cast<long>(x2 + (100480L*x0)), to_float_mask(tmp12));
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp27())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp27(), to_float_mask(tmp12));
                            auto tmp30 = decltype(tmp29)::blendv(tmp10, tmp29, tmp16);
                            auto tmp31 = tmp26 + tmp30;
                            auto tmp32 = tmp31 * tmp19;
                            auto tmp34 = tmp32 * tmp33;
                            tmp_acc0_vec = tmp_acc0_vec + tmp20;
                            tmp_acc1_vec = tmp_acc1_vec + tmp34;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr1[static_cast<long>(x1 + (784L*x0))] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr2[static_cast<long>(x1 + (784L*x0))] = static_cast<float>(tmp_acc1);
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (128L*x2) + (100352L*x1)));
                            auto tmp0 = c10::convert<int>(1L + x2);
                            auto tmp1 = static_cast<int>(1);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = masked_load(in_ptr2 + static_cast<long>(128L + x0 + (128L*x2) + (100480L*x1)), to_float_mask(tmp2));
                                auto tmp5 = masked_load(in_ptr3 + static_cast<long>(x0 + (128L*x2) + (100352L*x1)), to_float_mask(tmp2));
                                auto tmp6 = tmp4 + tmp5;
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                            auto tmp8 = static_cast<float>(0.0);
                            auto tmp9 = to_float_mask(tmp2);
                            auto tmp10 = at::vec::Vectorized<float>(tmp8);
                            auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                            auto tmp12 = tmp0 < tmp1;
                            auto tmp13 = [&]
                            {
                                auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x0 + (100480L*x1)), to_float_mask(tmp12));
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                            auto tmp16 = to_float_mask(tmp12);
                            auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                            auto tmp18 = tmp11 + tmp17;
                            auto tmp20 = tmp18 * tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr2 + static_cast<long>(128L + x0 + (128L*x2) + (100480L*x1)), to_float_mask(tmp2));
                                auto tmp23 = masked_load(in_ptr3 + static_cast<long>(x0 + (128L*x2) + (100352L*x1)), to_float_mask(tmp2));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp2));
                            auto tmp26 = decltype(tmp25)::blendv(tmp10, tmp25, tmp9);
                            auto tmp27 = [&]
                            {
                                auto tmp28 = masked_load(in_ptr2 + static_cast<long>(x0 + (100480L*x1)), to_float_mask(tmp12));
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp27())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp27(), to_float_mask(tmp12));
                            auto tmp30 = decltype(tmp29)::blendv(tmp10, tmp29, tmp16);
                            auto tmp31 = tmp26 + tmp30;
                            tmp_acc0_vec = tmp_acc0_vec + tmp20;
                            tmp_acc1_vec = tmp_acc1_vec + tmp31;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr6[static_cast<long>(x1 + (784L*x0))];
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp25 = out_ptr1[static_cast<long>(x1 + (784L*x0))];
                        auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (128L*x1) + (100352L*x0)));
                        auto tmp29 = out_ptr2[static_cast<long>(x1 + (784L*x0))];
                        auto tmp1 = c10::convert<int>(1L + x1);
                        auto tmp2 = static_cast<int>(1);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = masked_load(in_ptr2 + static_cast<long>(128L + x2 + (128L*x1) + (100480L*x0)), to_float_mask(tmp3));
                            auto tmp6 = masked_load(in_out_ptr2 + static_cast<long>(x2 + (128L*x1) + (100352L*x0)), to_float_mask(tmp3));
                            auto tmp7 = tmp5 + tmp6;
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp4())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp4(), to_float_mask(tmp3));
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = to_float_mask(tmp3);
                        auto tmp11 = at::vec::Vectorized<float>(tmp9);
                        auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                        auto tmp13 = tmp1 < tmp2;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = masked_load(in_ptr2 + static_cast<long>(x2 + (100480L*x0)), to_float_mask(tmp13));
                            return tmp15;
                        }
                        ;
                        auto tmp16 = decltype(tmp14())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp14(), to_float_mask(tmp13));
                        auto tmp17 = to_float_mask(tmp13);
                        auto tmp18 = decltype(tmp16)::blendv(tmp11, tmp16, tmp17);
                        auto tmp19 = tmp12 + tmp18;
                        auto tmp21 = tmp19 * tmp20;
                        auto tmp22 = static_cast<float>(128.0);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = at::vec::Vectorized<float>(tmp25);
                        auto tmp27 = tmp24 - tmp26;
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        auto tmp32 = tmp27 - tmp31;
                        auto tmp33 = at::vec::Vectorized<float>(tmp0);
                        auto tmp34 = tmp33 * tmp32;
                        tmp34.store(in_out_ptr2 + static_cast<long>(x2 + (128L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_slice_backward_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(static_cast<long>(x0) % static_cast<long>(3137L));
                    auto tmp1 = static_cast<int>(1);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = masked_load(in_ptr0 + static_cast<long>((-64L) + x1 + (64L*(static_cast<long>(x0) % static_cast<long>(3137L))) + (200704L*(c10::div_floor_integer(x0, 3137L)))), to_float_mask(tmp2));
                        return tmp4;
                    }
                    ;
                    auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = to_float_mask(tmp2);
                    auto tmp8 = at::vec::Vectorized<float>(tmp6);
                    auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp7);
                    tmp9.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12849152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_slice_backward_sum_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25096L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp10 = in_ptr5[static_cast<long>(x1 + (3137L*x0))];
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp17 = out_ptr1[static_cast<long>(x1 + (3137L*x0))];
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                        auto tmp21 = out_ptr2[static_cast<long>(x1 + (3137L*x0))];
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr4 + static_cast<long>((-64L) + x2 + (64L*x1) + (200704L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = to_float_mask(tmp2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp6);
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp7);
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp14 = static_cast<float>(64.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp22 = at::vec::Vectorized<float>(tmp21);
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp19 - tmp23;
                        auto tmp25 = at::vec::Vectorized<float>(tmp10);
                        auto tmp26 = tmp25 * tmp24;
                        auto tmp27 = tmp9 + tmp26;
                        tmp27.store(in_out_ptr0 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_sum_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(40L + x2 + (192L*x1) + (602304L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr1 + static_cast<long>(104L + x2 + (64L*x1) + (200768L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr1 + static_cast<long>(x2 + (24L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (192L*x1) + (602304L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr0 + static_cast<long>(80L + x2 + (64L*x1) + (200768L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (24L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (602304L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr0 + static_cast<long>(64L + x2 + (64L*x1) + (200768L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (50176L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (8L*x1) + (64L*x2) + (200768L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_stack_view_64 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(3136L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (3137L*x2) + (25096L*x1) + (200768L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x2) + (64L*x3) + (64L*x3_inner) + (200768L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(3136L); x3<static_cast<long>(3137L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (3137L*x2) + (25096L*x1) + (200768L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x2) + (64L*x3) + (200768L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (8L*x1) + (64L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x0);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x2);
                                auto tmp7 = static_cast<long>(1);
                                auto tmp8 = tmp6 >= tmp7;
                                auto tmp9 = [&]
                                {
                                    auto tmp10 = c10::convert<long>(x2);
                                    auto tmp11 = static_cast<long>(0);
                                    auto tmp12 = tmp10 >= tmp11;
                                    auto tmp13 = [&]
                                    {
                                        auto tmp14 = in_ptr2[static_cast<long>(x3 + (8L*x1) + (64L*x2) + (200768L*x0))];
                                        return tmp14;
                                    }
                                    ;
                                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                    auto tmp16 = in_ptr3[static_cast<long>(x3 + (8L*x1) + (64L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (200704L*x0))];
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                                auto tmp19 = static_cast<float>(0.0);
                                auto tmp20 = tmp8 ? tmp18 : tmp19;
                                auto tmp21 = in_ptr4[static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0))];
                                auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp24 = tmp0 >= tmp3;
                            auto tmp25 = static_cast<long>(16);
                            auto tmp26 = tmp0 < tmp25;
                            auto tmp27 = tmp24 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr0[static_cast<long>((-1606144L) + x2 + (3137L*x3) + (25096L*x1) + (200768L*x0))];
                                auto tmp30 = in_ptr1[static_cast<long>((-1606144L) + x1 + (8L*x3) + (64L*x2) + (200768L*x0))];
                                auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                                auto tmp32 = out_ptr0[static_cast<long>((-512L) + x3 + (8L*x1) + (64L*x0))];
                                auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                                auto tmp34 = decltype(tmp31)(tmp31 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp36 = tmp0 >= tmp25;
                            auto tmp37 = static_cast<long>(24);
                            auto tmp38 = tmp0 < tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x2);
                                auto tmp41 = static_cast<long>(1);
                                auto tmp42 = tmp40 >= tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = c10::convert<long>(x3 + (8L*x1));
                                    auto tmp45 = static_cast<long>(0);
                                    auto tmp46 = tmp44 >= tmp45;
                                    auto tmp47 = static_cast<long>(16);
                                    auto tmp48 = tmp44 < tmp47;
                                    auto tmp49 = [&]
                                    {
                                        auto tmp50 = in_ptr5[static_cast<long>((-802816L) + x3 + (8L*x1) + (16L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (50176L*x0))];
                                        return tmp50;
                                    }
                                    ;
                                    auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                    auto tmp52 = tmp44 >= tmp47;
                                    auto tmp53 = static_cast<long>(40);
                                    auto tmp54 = tmp44 < tmp53;
                                    auto tmp55 = tmp52 & tmp54;
                                    auto tmp56 = [&]
                                    {
                                        auto tmp57 = in_ptr6[static_cast<long>((-1204240L) + x3 + (8L*x1) + (24L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (75264L*x0))];
                                        return tmp57;
                                    }
                                    ;
                                    auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                                    auto tmp59 = tmp44 >= tmp53;
                                    auto tmp60 = static_cast<long>(64);
                                    auto tmp61 = tmp44 < tmp60;
                                    auto tmp62 = [&]
                                    {
                                        auto tmp63 = in_ptr7[static_cast<long>((-1204264L) + x3 + (8L*x1) + (24L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (75264L*x0))];
                                        return tmp63;
                                    }
                                    ;
                                    auto tmp64 = tmp59 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                                    auto tmp65 = tmp55 ? tmp58 : tmp64;
                                    auto tmp66 = tmp48 ? tmp51 : tmp65;
                                    return tmp66;
                                }
                                ;
                                auto tmp67 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp68 = static_cast<float>(0.0);
                                auto tmp69 = tmp42 ? tmp67 : tmp68;
                                auto tmp70 = in_ptr8[static_cast<long>((-3212288L) + x3 + (8L*x2) + (25096L*x1) + (200768L*x0))];
                                auto tmp71 = decltype(tmp69)(tmp69 + tmp70);
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp36 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp73 = tmp27 ? tmp35 : tmp72;
                            auto tmp74 = tmp4 ? tmp23 : tmp73;
                            out_ptr1[static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0))] = tmp74;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>((8L*(static_cast<long>(x0) % static_cast<long>(3137L))) + (25096L*(static_cast<long>(c10::div_floor_integer((x1 + x1_inner), 8L)) % static_cast<long>(8L))) + (200768L*(c10::div_floor_integer(x0, 3137L))) + (1606144L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(8L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp13 = in_ptr4[static_cast<long>(x0)];
                    auto tmp18 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(64.0);
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
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_slice_backward_66 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr0 + static_cast<long>(64L + x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200768L*x0)), to_float_mask(tmp2));
                            auto tmp5 = masked_load(in_ptr1 + static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0)), to_float_mask(tmp2));
                            auto tmp6 = tmp4 + tmp5;
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = to_float_mask(tmp2);
                        auto tmp10 = at::vec::Vectorized<float>(tmp8);
                        auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = masked_load(in_ptr0 + static_cast<long>(x2 + (200768L*x0)), to_float_mask(tmp12));
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                        auto tmp16 = to_float_mask(tmp12);
                        auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                        auto tmp18 = tmp11 + tmp17;
                        tmp18.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12849152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_68 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25096L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(64.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_sum_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(40L + x2 + (192L*x1) + (602304L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr1 + static_cast<long>(104L + x2 + (64L*x1) + (200768L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr1 + static_cast<long>(x2 + (24L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1176L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(16L + x2 + (192L*x1) + (602304L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr2 + static_cast<long>(80L + x2 + (64L*x1) + (200768L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (24L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(600L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (192L*x1) + (602304L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr2 + static_cast<long>(64L + x2 + (64L*x1) + (200768L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (50176L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_mul_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (8L*x1) + (64L*x2) + (200768L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_stack_view_73 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(3136L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (3137L*x2) + (25096L*x1) + (200768L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x2) + (64L*x3) + (64L*x3_inner) + (200768L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(3136L); x3<static_cast<long>(3137L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (3137L*x2) + (25096L*x1) + (200768L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x2) + (64L*x3) + (200768L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (8L*x1) + (64L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x0);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x2);
                                auto tmp7 = static_cast<long>(1);
                                auto tmp8 = tmp6 >= tmp7;
                                auto tmp9 = [&]
                                {
                                    auto tmp10 = c10::convert<long>(x2);
                                    auto tmp11 = static_cast<long>(0);
                                    auto tmp12 = tmp10 >= tmp11;
                                    auto tmp13 = [&]
                                    {
                                        auto tmp14 = in_ptr2[static_cast<long>(x3 + (8L*x1) + (64L*x2) + (200768L*x0))];
                                        return tmp14;
                                    }
                                    ;
                                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                    auto tmp16 = in_ptr3[static_cast<long>(x3 + (8L*x1) + (64L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (200704L*x0))];
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                                auto tmp19 = static_cast<float>(0.0);
                                auto tmp20 = tmp8 ? tmp18 : tmp19;
                                auto tmp21 = in_ptr4[static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0))];
                                auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp24 = tmp0 >= tmp3;
                            auto tmp25 = static_cast<long>(16);
                            auto tmp26 = tmp0 < tmp25;
                            auto tmp27 = tmp24 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr0[static_cast<long>((-1606144L) + x2 + (3137L*x3) + (25096L*x1) + (200768L*x0))];
                                auto tmp30 = in_ptr1[static_cast<long>((-1606144L) + x1 + (8L*x3) + (64L*x2) + (200768L*x0))];
                                auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                                auto tmp32 = out_ptr0[static_cast<long>((-512L) + x3 + (8L*x1) + (64L*x0))];
                                auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                                auto tmp34 = decltype(tmp31)(tmp31 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp36 = tmp0 >= tmp25;
                            auto tmp37 = static_cast<long>(24);
                            auto tmp38 = tmp0 < tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x2);
                                auto tmp41 = static_cast<long>(1);
                                auto tmp42 = tmp40 >= tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = c10::convert<long>(x3 + (8L*x1));
                                    auto tmp45 = static_cast<long>(0);
                                    auto tmp46 = tmp44 >= tmp45;
                                    auto tmp47 = static_cast<long>(16);
                                    auto tmp48 = tmp44 < tmp47;
                                    auto tmp49 = [&]
                                    {
                                        auto tmp50 = in_ptr5[static_cast<long>((-802816L) + x3 + (8L*x1) + (16L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (50176L*x0))];
                                        return tmp50;
                                    }
                                    ;
                                    auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                    auto tmp52 = tmp44 >= tmp47;
                                    auto tmp53 = static_cast<long>(40);
                                    auto tmp54 = tmp44 < tmp53;
                                    auto tmp55 = tmp52 & tmp54;
                                    auto tmp56 = [&]
                                    {
                                        auto tmp57 = in_ptr6[static_cast<long>((-1204240L) + x3 + (8L*x1) + (24L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (75264L*x0))];
                                        return tmp57;
                                    }
                                    ;
                                    auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                                    auto tmp59 = tmp44 >= tmp53;
                                    auto tmp60 = static_cast<long>(64);
                                    auto tmp61 = tmp44 < tmp60;
                                    auto tmp62 = [&]
                                    {
                                        auto tmp63 = in_ptr7[static_cast<long>((-1204264L) + x3 + (8L*x1) + (24L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (75264L*x0))];
                                        return tmp63;
                                    }
                                    ;
                                    auto tmp64 = tmp59 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                                    auto tmp65 = tmp55 ? tmp58 : tmp64;
                                    auto tmp66 = tmp48 ? tmp51 : tmp65;
                                    return tmp66;
                                }
                                ;
                                auto tmp67 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp68 = static_cast<float>(0.0);
                                auto tmp69 = tmp42 ? tmp67 : tmp68;
                                auto tmp70 = in_ptr8[static_cast<long>((-3212288L) + x3 + (8L*x2) + (25096L*x1) + (200768L*x0))];
                                auto tmp71 = decltype(tmp69)(tmp69 + tmp70);
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp36 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp73 = tmp27 ? tmp35 : tmp72;
                            auto tmp74 = tmp4 ? tmp23 : tmp73;
                            out_ptr1[static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0))] = tmp74;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>((8L*(static_cast<long>(x0) % static_cast<long>(3137L))) + (25096L*(static_cast<long>(c10::div_floor_integer((x1 + x1_inner), 8L)) % static_cast<long>(8L))) + (200768L*(c10::div_floor_integer(x0, 3137L))) + (1606144L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(8L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp13 = in_ptr4[static_cast<long>(x0)];
                    auto tmp18 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(64.0);
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
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_layer_norm_backward_sum_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr3 = in_out_ptr2;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = static_cast<int>(0);
                    auto tmp1 = static_cast<int>(1);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = masked_load(in_ptr2 + static_cast<long>(200704L + x0 + (200768L*x1)), to_float_mask(tmp2));
                        auto tmp5 = masked_load(in_ptr3 + static_cast<long>(200640L + x0 + (200704L*x1)), to_float_mask(tmp2));
                        auto tmp6 = tmp4 + tmp5;
                        return tmp6;
                    }
                    ;
                    auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp9 = to_float_mask(tmp2);
                    auto tmp10 = at::vec::Vectorized<float>(tmp8);
                    auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                    auto tmp12 = tmp0 < tmp1;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x0 + (200768L*x1)), to_float_mask(tmp12));
                        return tmp14;
                    }
                    ;
                    auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                    auto tmp16 = to_float_mask(tmp12);
                    auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                    auto tmp18 = tmp11 + tmp17;
                    tmp_acc0_vec = tmp_acc0_vec + tmp18;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (64L*x1) + (200704L*x0)));
                            auto tmp0 = c10::convert<int>(1L + x1);
                            auto tmp1 = static_cast<int>(1);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = masked_load(in_ptr2 + static_cast<long>(64L + x2 + (64L*x1) + (200768L*x0)), to_float_mask(tmp2));
                                auto tmp5 = masked_load(in_ptr3 + static_cast<long>(x2 + (64L*x1) + (200704L*x0)), to_float_mask(tmp2));
                                auto tmp6 = tmp4 + tmp5;
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                            auto tmp8 = static_cast<float>(0.0);
                            auto tmp9 = to_float_mask(tmp2);
                            auto tmp10 = at::vec::Vectorized<float>(tmp8);
                            auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                            auto tmp12 = tmp0 < tmp1;
                            auto tmp13 = [&]
                            {
                                auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x2 + (200768L*x0)), to_float_mask(tmp12));
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                            auto tmp16 = to_float_mask(tmp12);
                            auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                            auto tmp18 = tmp11 + tmp17;
                            auto tmp20 = tmp18 * tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr2 + static_cast<long>(64L + x2 + (64L*x1) + (200768L*x0)), to_float_mask(tmp2));
                                auto tmp23 = masked_load(in_ptr3 + static_cast<long>(x2 + (64L*x1) + (200704L*x0)), to_float_mask(tmp2));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp2));
                            auto tmp26 = decltype(tmp25)::blendv(tmp10, tmp25, tmp9);
                            auto tmp27 = [&]
                            {
                                auto tmp28 = masked_load(in_ptr2 + static_cast<long>(x2 + (200768L*x0)), to_float_mask(tmp12));
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp27())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp27(), to_float_mask(tmp12));
                            auto tmp30 = decltype(tmp29)::blendv(tmp10, tmp29, tmp16);
                            auto tmp31 = tmp26 + tmp30;
                            auto tmp32 = tmp31 * tmp19;
                            auto tmp34 = tmp32 * tmp33;
                            tmp_acc0_vec = tmp_acc0_vec + tmp20;
                            tmp_acc1_vec = tmp_acc1_vec + tmp34;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr1[static_cast<long>(x1 + (3136L*x0))] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr2[static_cast<long>(x1 + (3136L*x0))] = static_cast<float>(tmp_acc1);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (64L*x2) + (200704L*x1)));
                            auto tmp0 = c10::convert<int>(1L + x2);
                            auto tmp1 = static_cast<int>(1);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = masked_load(in_ptr2 + static_cast<long>(64L + x0 + (64L*x2) + (200768L*x1)), to_float_mask(tmp2));
                                auto tmp5 = masked_load(in_ptr3 + static_cast<long>(x0 + (64L*x2) + (200704L*x1)), to_float_mask(tmp2));
                                auto tmp6 = tmp4 + tmp5;
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                            auto tmp8 = static_cast<float>(0.0);
                            auto tmp9 = to_float_mask(tmp2);
                            auto tmp10 = at::vec::Vectorized<float>(tmp8);
                            auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp9);
                            auto tmp12 = tmp0 < tmp1;
                            auto tmp13 = [&]
                            {
                                auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x0 + (200768L*x1)), to_float_mask(tmp12));
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp12));
                            auto tmp16 = to_float_mask(tmp12);
                            auto tmp17 = decltype(tmp15)::blendv(tmp10, tmp15, tmp16);
                            auto tmp18 = tmp11 + tmp17;
                            auto tmp20 = tmp18 * tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr2 + static_cast<long>(64L + x0 + (64L*x2) + (200768L*x1)), to_float_mask(tmp2));
                                auto tmp23 = masked_load(in_ptr3 + static_cast<long>(x0 + (64L*x2) + (200704L*x1)), to_float_mask(tmp2));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp2));
                            auto tmp26 = decltype(tmp25)::blendv(tmp10, tmp25, tmp9);
                            auto tmp27 = [&]
                            {
                                auto tmp28 = masked_load(in_ptr2 + static_cast<long>(x0 + (200768L*x1)), to_float_mask(tmp12));
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp27())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp27(), to_float_mask(tmp12));
                            auto tmp30 = decltype(tmp29)::blendv(tmp10, tmp29, tmp16);
                            auto tmp31 = tmp26 + tmp30;
                            tmp_acc0_vec = tmp_acc0_vec + tmp20;
                            tmp_acc1_vec = tmp_acc1_vec + tmp31;
                        }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr6[static_cast<long>(x1 + (3136L*x0))];
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp25 = out_ptr1[static_cast<long>(x1 + (3136L*x0))];
                        auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (64L*x1) + (200704L*x0)));
                        auto tmp29 = out_ptr2[static_cast<long>(x1 + (3136L*x0))];
                        auto tmp1 = c10::convert<int>(1L + x1);
                        auto tmp2 = static_cast<int>(1);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = masked_load(in_ptr2 + static_cast<long>(64L + x2 + (64L*x1) + (200768L*x0)), to_float_mask(tmp3));
                            auto tmp6 = masked_load(in_out_ptr2 + static_cast<long>(x2 + (64L*x1) + (200704L*x0)), to_float_mask(tmp3));
                            auto tmp7 = tmp5 + tmp6;
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp4())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp4(), to_float_mask(tmp3));
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = to_float_mask(tmp3);
                        auto tmp11 = at::vec::Vectorized<float>(tmp9);
                        auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                        auto tmp13 = tmp1 < tmp2;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = masked_load(in_ptr2 + static_cast<long>(x2 + (200768L*x0)), to_float_mask(tmp13));
                            return tmp15;
                        }
                        ;
                        auto tmp16 = decltype(tmp14())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp14(), to_float_mask(tmp13));
                        auto tmp17 = to_float_mask(tmp13);
                        auto tmp18 = decltype(tmp16)::blendv(tmp11, tmp16, tmp17);
                        auto tmp19 = tmp12 + tmp18;
                        auto tmp21 = tmp19 * tmp20;
                        auto tmp22 = static_cast<float>(64.0);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = at::vec::Vectorized<float>(tmp25);
                        auto tmp27 = tmp24 - tmp26;
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        auto tmp32 = tmp27 - tmp31;
                        auto tmp33 = at::vec::Vectorized<float>(tmp0);
                        auto tmp34 = tmp33 * tmp32;
                        tmp34.store(in_out_ptr2 + static_cast<long>(x2 + (64L*x1) + (200704L*x0)));
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
    primals_2, primals_4, primals_6, primals_8, primals_11, primals_13, primals_15, primals_17, primals_20, primals_22, primals_24, primals_26, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_67, primals_69, primals_71, primals_75, primals_77, primals_79, primals_95, primals_97, primals_99, primals_103, primals_105, primals_107, primals_123, primals_125, primals_127, primals_131, primals_133, primals_135, primals_153, mul, view_1, cat_1, getitem_3, rsqrt_1, view_3, slice_8, getitem_7, getitem_8, getitem_9, cat_2, view_15, mul_6, view_17, addmm_2, view_19, view_21, cat_3, getitem_13, rsqrt_3, view_23, slice_20, getitem_17, getitem_18, getitem_19, cat_4, view_35, mul_15, view_37, addmm_6, view_39, clone_15, mul_20, view_43, cat_6, getitem_25, rsqrt_6, view_45, slice_35, getitem_29, getitem_30, getitem_31, cat_7, view_57, mul_26, view_59, addmm_10, view_61, view_63, cat_8, getitem_35, rsqrt_8, view_65, slice_47, getitem_39, getitem_40, getitem_41, cat_9, view_77, mul_35, view_79, addmm_14, view_81, clone_31, mul_40, view_85, cat_11, getitem_47, rsqrt_11, view_87, slice_62, getitem_51, getitem_52, getitem_53, cat_12, view_99, mul_46, view_101, addmm_18, view_103, view_105, cat_13, getitem_57, rsqrt_13, view_107, slice_74, getitem_61, getitem_62, getitem_63, cat_14, view_119, mul_55, view_121, addmm_22, view_123, clone_47, mul_60, view_127, cat_16, getitem_69, rsqrt_16, view_129, slice_89, getitem_73, getitem_74, getitem_75, cat_17, view_141, mul_66, view_143, addmm_26, view_145, view_147, cat_18, getitem_79, rsqrt_18, view_149, slice_101, getitem_83, getitem_84, getitem_85, cat_19, view_161, mul_75, view_163, addmm_30, view_165, mul_80, clone_64, permute_97, div_8, permute_101, permute_105, div_9, permute_109, permute_116, permute_117, permute_118, permute_119, alias_8, permute_122, permute_128, permute_132, div_11, permute_136, permute_143, permute_144, permute_145, permute_146, alias_9, permute_149, div_13, permute_157, permute_161, div_14, permute_165, permute_172, permute_173, permute_174, permute_175, alias_10, permute_178, permute_184, permute_188, div_16, permute_192, permute_199, permute_200, permute_201, permute_202, alias_11, permute_205, div_18, permute_213, permute_217, div_19, permute_221, permute_228, permute_229, permute_230, permute_231, alias_12, permute_234, permute_240, permute_244, div_21, permute_248, permute_255, permute_256, permute_257, permute_258, alias_13, permute_261, div_23, permute_269, permute_273, div_24, permute_277, permute_284, permute_285, permute_286, permute_287, alias_14, permute_290, permute_296, permute_300, div_26, permute_304, permute_311, permute_312, permute_313, permute_314, alias_15, permute_317, div_28, tangents_1 = args
    args.clear()
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_20, (320, ), (1, ))
    assert_size_stride(primals_22, (320, ), (1, ))
    assert_size_stride(primals_24, (320, ), (1, ))
    assert_size_stride(primals_26, (320, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_39, (64, 3, 4, 4), (48, 1, 12, 3))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_43, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_47, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_49, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_51, (24, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_67, (128, 64, 2, 2), (256, 1, 128, 64))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_71, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_75, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_77, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_79, (48, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_95, (320, 128, 2, 2), (512, 1, 256, 128))
    assert_size_stride(primals_97, (320, ), (1, ))
    assert_size_stride(primals_99, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_103, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_105, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_107, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_123, (512, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_127, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_131, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_135, (192, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_153, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(mul, (8, 3136, 64), (200704, 64, 1))
    assert_size_stride(view_1, (8, 64, 56, 56), (200768, 1, 3584, 64))
    assert_size_stride(cat_1, (8, 3137, 64), (200768, 64, 1))
    assert_size_stride(getitem_3, (8, 3137, 1), (3137, 1, 1))
    assert_size_stride(rsqrt_1, (8, 3137, 1), (3137, 1, 1))
    assert_size_stride(view_3, (25096, 64), (64, 1))
    assert_size_stride(slice_8, (8, 8, 3136, 8), (602304, 8, 192, 1))
    assert_size_stride(getitem_7, (8, 16, 56, 56), (602304, 1, 10752, 192))
    assert_size_stride(getitem_8, (8, 24, 56, 56), (602304, 1, 10752, 192))
    assert_size_stride(getitem_9, (8, 24, 56, 56), (602304, 1, 10752, 192))
    assert_size_stride(cat_2, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(view_15, (25096, 64), (64, 1))
    assert_size_stride(mul_6, (8, 3137, 64), (200768, 64, 1))
    assert_size_stride(view_17, (25096, 64), (64, 1))
    assert_size_stride(addmm_2, (25096, 512), (512, 1))
    assert_size_stride(view_19, (25096, 512), (512, 1))
    assert_size_stride(view_21, (8, 64, 56, 56), (200768, 1, 3584, 64))
    assert_size_stride(cat_3, (8, 3137, 64), (200768, 64, 1))
    assert_size_stride(getitem_13, (8, 3137, 1), (3137, 1, 1))
    assert_size_stride(rsqrt_3, (8, 3137, 1), (3137, 1, 1))
    assert_size_stride(view_23, (25096, 64), (64, 1))
    assert_size_stride(slice_20, (8, 8, 3136, 8), (602304, 8, 192, 1))
    assert_size_stride(getitem_17, (8, 16, 56, 56), (602304, 1, 10752, 192))
    assert_size_stride(getitem_18, (8, 24, 56, 56), (602304, 1, 10752, 192))
    assert_size_stride(getitem_19, (8, 24, 56, 56), (602304, 1, 10752, 192))
    assert_size_stride(cat_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(view_35, (25096, 64), (64, 1))
    assert_size_stride(mul_15, (8, 3137, 64), (200768, 64, 1))
    assert_size_stride(view_37, (25096, 64), (64, 1))
    assert_size_stride(addmm_6, (25096, 512), (512, 1))
    assert_size_stride(view_39, (25096, 512), (512, 1))
    assert_size_stride(clone_15, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(mul_20, (8, 784, 128), (100352, 128, 1))
    assert_size_stride(view_43, (8, 128, 28, 28), (100480, 1, 3584, 128))
    assert_size_stride(cat_6, (8, 785, 128), (100480, 128, 1))
    assert_size_stride(getitem_25, (8, 785, 1), (785, 1, 1))
    assert_size_stride(rsqrt_6, (8, 785, 1), (785, 1, 1))
    assert_size_stride(view_45, (6280, 128), (128, 1))
    assert_size_stride(slice_35, (8, 8, 784, 16), (301440, 16, 384, 1))
    assert_size_stride(getitem_29, (8, 32, 28, 28), (301440, 1, 10752, 384))
    assert_size_stride(getitem_30, (8, 48, 28, 28), (301440, 1, 10752, 384))
    assert_size_stride(getitem_31, (8, 48, 28, 28), (301440, 1, 10752, 384))
    assert_size_stride(cat_7, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(view_57, (6280, 128), (128, 1))
    assert_size_stride(mul_26, (8, 785, 128), (100480, 128, 1))
    assert_size_stride(view_59, (6280, 128), (128, 1))
    assert_size_stride(addmm_10, (6280, 1024), (1024, 1))
    assert_size_stride(view_61, (6280, 1024), (1024, 1))
    assert_size_stride(view_63, (8, 128, 28, 28), (100480, 1, 3584, 128))
    assert_size_stride(cat_8, (8, 785, 128), (100480, 128, 1))
    assert_size_stride(getitem_35, (8, 785, 1), (785, 1, 1))
    assert_size_stride(rsqrt_8, (8, 785, 1), (785, 1, 1))
    assert_size_stride(view_65, (6280, 128), (128, 1))
    assert_size_stride(slice_47, (8, 8, 784, 16), (301440, 16, 384, 1))
    assert_size_stride(getitem_39, (8, 32, 28, 28), (301440, 1, 10752, 384))
    assert_size_stride(getitem_40, (8, 48, 28, 28), (301440, 1, 10752, 384))
    assert_size_stride(getitem_41, (8, 48, 28, 28), (301440, 1, 10752, 384))
    assert_size_stride(cat_9, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(view_77, (6280, 128), (128, 1))
    assert_size_stride(mul_35, (8, 785, 128), (100480, 128, 1))
    assert_size_stride(view_79, (6280, 128), (128, 1))
    assert_size_stride(addmm_14, (6280, 1024), (1024, 1))
    assert_size_stride(view_81, (6280, 1024), (1024, 1))
    assert_size_stride(clone_31, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_40, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_85, (8, 320, 14, 14), (63040, 1, 4480, 320))
    assert_size_stride(cat_11, (8, 197, 320), (63040, 320, 1))
    assert_size_stride(getitem_47, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_11, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_87, (1576, 320), (320, 1))
    assert_size_stride(slice_62, (8, 8, 196, 40), (189120, 40, 960, 1))
    assert_size_stride(getitem_51, (8, 80, 14, 14), (189120, 1, 13440, 960))
    assert_size_stride(getitem_52, (8, 120, 14, 14), (189120, 1, 13440, 960))
    assert_size_stride(getitem_53, (8, 120, 14, 14), (189120, 1, 13440, 960))
    assert_size_stride(cat_12, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(view_99, (1576, 320), (320, 1))
    assert_size_stride(mul_46, (8, 197, 320), (63040, 320, 1))
    assert_size_stride(view_101, (1576, 320), (320, 1))
    assert_size_stride(addmm_18, (1576, 1280), (1280, 1))
    assert_size_stride(view_103, (1576, 1280), (1280, 1))
    assert_size_stride(view_105, (8, 320, 14, 14), (63040, 1, 4480, 320))
    assert_size_stride(cat_13, (8, 197, 320), (63040, 320, 1))
    assert_size_stride(getitem_57, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_13, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_107, (1576, 320), (320, 1))
    assert_size_stride(slice_74, (8, 8, 196, 40), (189120, 40, 960, 1))
    assert_size_stride(getitem_61, (8, 80, 14, 14), (189120, 1, 13440, 960))
    assert_size_stride(getitem_62, (8, 120, 14, 14), (189120, 1, 13440, 960))
    assert_size_stride(getitem_63, (8, 120, 14, 14), (189120, 1, 13440, 960))
    assert_size_stride(cat_14, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(view_119, (1576, 320), (320, 1))
    assert_size_stride(mul_55, (8, 197, 320), (63040, 320, 1))
    assert_size_stride(view_121, (1576, 320), (320, 1))
    assert_size_stride(addmm_22, (1576, 1280), (1280, 1))
    assert_size_stride(view_123, (1576, 1280), (1280, 1))
    assert_size_stride(clone_47, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_60, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_127, (8, 512, 7, 7), (25600, 1, 3584, 512))
    assert_size_stride(cat_16, (8, 50, 512), (25600, 512, 1))
    assert_size_stride(getitem_69, (8, 50, 1), (50, 1, 1))
    assert_size_stride(rsqrt_16, (8, 50, 1), (50, 1, 1))
    assert_size_stride(view_129, (400, 512), (512, 1))
    assert_size_stride(slice_89, (8, 8, 49, 64), (76800, 64, 1536, 1))
    assert_size_stride(getitem_73, (8, 128, 7, 7), (76800, 1, 10752, 1536))
    assert_size_stride(getitem_74, (8, 192, 7, 7), (76800, 1, 10752, 1536))
    assert_size_stride(getitem_75, (8, 192, 7, 7), (76800, 1, 10752, 1536))
    assert_size_stride(cat_17, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(view_141, (400, 512), (512, 1))
    assert_size_stride(mul_66, (8, 50, 512), (25600, 512, 1))
    assert_size_stride(view_143, (400, 512), (512, 1))
    assert_size_stride(addmm_26, (400, 2048), (2048, 1))
    assert_size_stride(view_145, (400, 2048), (2048, 1))
    assert_size_stride(view_147, (8, 512, 7, 7), (25600, 1, 3584, 512))
    assert_size_stride(cat_18, (8, 50, 512), (25600, 512, 1))
    assert_size_stride(getitem_79, (8, 50, 1), (50, 1, 1))
    assert_size_stride(rsqrt_18, (8, 50, 1), (50, 1, 1))
    assert_size_stride(view_149, (400, 512), (512, 1))
    assert_size_stride(slice_101, (8, 8, 49, 64), (76800, 64, 1536, 1))
    assert_size_stride(getitem_83, (8, 128, 7, 7), (76800, 1, 10752, 1536))
    assert_size_stride(getitem_84, (8, 192, 7, 7), (76800, 1, 10752, 1536))
    assert_size_stride(getitem_85, (8, 192, 7, 7), (76800, 1, 10752, 1536))
    assert_size_stride(cat_19, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(view_161, (400, 512), (512, 1))
    assert_size_stride(mul_75, (8, 50, 512), (25600, 512, 1))
    assert_size_stride(view_163, (400, 512), (512, 1))
    assert_size_stride(addmm_30, (400, 2048), (2048, 1))
    assert_size_stride(view_165, (400, 2048), (2048, 1))
    assert_size_stride(mul_80, (8, 50, 512), (25600, 512, 1))
    assert_size_stride(clone_64, (8, 512), (512, 1))
    assert_size_stride(permute_97, (1000, 512), (512, 1))
    assert_size_stride(div_8, (8, 50, 1), (50, 1, 1))
    assert_size_stride(permute_101, (512, 2048), (2048, 1))
    assert_size_stride(permute_105, (2048, 512), (512, 1))
    assert_size_stride(div_9, (8, 50, 1), (50, 1, 1))
    assert_size_stride(permute_109, (512, 512), (512, 1))
    assert_size_stride(permute_116, (64, 64, 50), (3200, 1, 64))
    assert_size_stride(permute_117, (64, 64, 64), (4096, 1, 64))
    assert_size_stride(permute_118, (64, 50, 64), (3200, 64, 1))
    assert_size_stride(permute_119, (64, 64, 50), (3200, 1, 64))
    assert_size_stride(alias_8, (8, 8, 50, 64), (25600, 1, 512, 8))
    assert_size_stride(permute_122, (1536, 512), (512, 1))
    assert_size_stride(permute_128, (512, 2048), (2048, 1))
    assert_size_stride(permute_132, (2048, 512), (512, 1))
    assert_size_stride(div_11, (8, 50, 1), (50, 1, 1))
    assert_size_stride(permute_136, (512, 512), (512, 1))
    assert_size_stride(permute_143, (64, 64, 50), (3200, 1, 64))
    assert_size_stride(permute_144, (64, 64, 64), (4096, 1, 64))
    assert_size_stride(permute_145, (64, 50, 64), (3200, 64, 1))
    assert_size_stride(permute_146, (64, 64, 50), (3200, 1, 64))
    assert_size_stride(alias_9, (8, 8, 50, 64), (25600, 1, 512, 8))
    assert_size_stride(permute_149, (1536, 512), (512, 1))
    assert_size_stride(div_13, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_157, (320, 1280), (1280, 1))
    assert_size_stride(permute_161, (1280, 320), (320, 1))
    assert_size_stride(div_14, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_165, (320, 320), (320, 1))
    assert_size_stride(permute_172, (64, 40, 197), (7880, 1, 40))
    assert_size_stride(permute_173, (64, 40, 40), (1600, 1, 40))
    assert_size_stride(permute_174, (64, 197, 40), (7880, 40, 1))
    assert_size_stride(permute_175, (64, 40, 197), (7880, 1, 40))
    assert_size_stride(alias_10, (8, 8, 197, 40), (63040, 1, 320, 8))
    assert_size_stride(permute_178, (960, 320), (320, 1))
    assert_size_stride(permute_184, (320, 1280), (1280, 1))
    assert_size_stride(permute_188, (1280, 320), (320, 1))
    assert_size_stride(div_16, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_192, (320, 320), (320, 1))
    assert_size_stride(permute_199, (64, 40, 197), (7880, 1, 40))
    assert_size_stride(permute_200, (64, 40, 40), (1600, 1, 40))
    assert_size_stride(permute_201, (64, 197, 40), (7880, 40, 1))
    assert_size_stride(permute_202, (64, 40, 197), (7880, 1, 40))
    assert_size_stride(alias_11, (8, 8, 197, 40), (63040, 1, 320, 8))
    assert_size_stride(permute_205, (960, 320), (320, 1))
    assert_size_stride(div_18, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_213, (128, 1024), (1024, 1))
    assert_size_stride(permute_217, (1024, 128), (128, 1))
    assert_size_stride(div_19, (8, 785, 1), (785, 1, 1))
    assert_size_stride(permute_221, (128, 128), (128, 1))
    assert_size_stride(permute_228, (64, 16, 785), (12560, 1, 16))
    assert_size_stride(permute_229, (64, 16, 16), (256, 1, 16))
    assert_size_stride(permute_230, (64, 785, 16), (12560, 16, 1))
    assert_size_stride(permute_231, (64, 16, 785), (12560, 1, 16))
    assert_size_stride(alias_12, (8, 8, 785, 16), (100480, 1, 128, 8))
    assert_size_stride(permute_234, (384, 128), (128, 1))
    assert_size_stride(permute_240, (128, 1024), (1024, 1))
    assert_size_stride(permute_244, (1024, 128), (128, 1))
    assert_size_stride(div_21, (8, 785, 1), (785, 1, 1))
    assert_size_stride(permute_248, (128, 128), (128, 1))
    assert_size_stride(permute_255, (64, 16, 785), (12560, 1, 16))
    assert_size_stride(permute_256, (64, 16, 16), (256, 1, 16))
    assert_size_stride(permute_257, (64, 785, 16), (12560, 16, 1))
    assert_size_stride(permute_258, (64, 16, 785), (12560, 1, 16))
    assert_size_stride(alias_13, (8, 8, 785, 16), (100480, 1, 128, 8))
    assert_size_stride(permute_261, (384, 128), (128, 1))
    assert_size_stride(div_23, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_269, (64, 512), (512, 1))
    assert_size_stride(permute_273, (512, 64), (64, 1))
    assert_size_stride(div_24, (8, 3137, 1), (3137, 1, 1))
    assert_size_stride(permute_277, (64, 64), (64, 1))
    assert_size_stride(permute_284, (64, 8, 3137), (25096, 1, 8))
    assert_size_stride(permute_285, (64, 8, 8), (64, 1, 8))
    assert_size_stride(permute_286, (64, 3137, 8), (25096, 8, 1))
    assert_size_stride(permute_287, (64, 8, 3137), (25096, 1, 8))
    assert_size_stride(alias_14, (8, 8, 3137, 8), (200768, 1, 64, 8))
    assert_size_stride(permute_290, (192, 64), (64, 1))
    assert_size_stride(permute_296, (64, 512), (512, 1))
    assert_size_stride(permute_300, (512, 64), (64, 1))
    assert_size_stride(div_26, (8, 3137, 1), (3137, 1, 1))
    assert_size_stride(permute_304, (64, 64), (64, 1))
    assert_size_stride(permute_311, (64, 8, 3137), (25096, 1, 8))
    assert_size_stride(permute_312, (64, 8, 8), (64, 1, 8))
    assert_size_stride(permute_313, (64, 3137, 8), (25096, 8, 1))
    assert_size_stride(permute_314, (64, 8, 3137), (25096, 1, 8))
    assert_size_stride(alias_15, (8, 8, 3137, 8), (200768, 1, 64, 8))
    assert_size_stride(permute_317, (192, 64), (64, 1))
    assert_size_stride(div_28, (8, 3136, 1), (3136, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_97, out=buf0)
    del permute_97
    buf1 = empty((1000, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_64, out=buf1)
    del clone_64
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((8, 50, 1), (50, 1, 400), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 50, 1), (50, 1, 400), device='cpu', dtype=torch.float32)
    buf5 = empty((8, 50, 512), device='cpu', dtype=torch.float32)
    buf6 = empty((512, ), device='cpu', dtype=torch.float32)
    buf7 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_select_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_8.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del div_8
    del mul_80
    del primals_37
    del tangents_1
    buf8 = empty((400, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (400, 512), (512, 1), 0), permute_101, out=buf8)
    del permute_101
    buf9 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (512, 400), (1, 512), 0), view_165, out=buf9)
    del view_165
    buf10 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf11 = reinterpret_tensor(buf8, (8, 50, 2048), (102400, 2048, 1), 0); del buf8  # reuse
    cpp_fused_gelu_gelu_backward_sum_1(c_void_p(buf11.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(addmm_30.data_ptr()), c_void_p(buf10.data_ptr()))
    del addmm_30
    buf12 = empty((400, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (400, 2048), (2048, 1), 0), permute_105, out=buf12)
    del permute_105
    buf13 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (2048, 400), (1, 2048), 0), view_163, out=buf13)
    del view_163
    buf14 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf15 = buf4; del buf4  # reuse
    buf16 = buf3; del buf3  # reuse
    buf17 = empty((512, ), device='cpu', dtype=torch.float32)
    buf18 = empty((512, ), device='cpu', dtype=torch.float32)
    buf19 = reinterpret_tensor(buf12, (8, 50, 512), (25600, 512, 1), 0); del buf12  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_2(c_void_p(buf19.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(mul_75.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(div_9.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    del div_9
    del mul_75
    del primals_35
    buf20 = reinterpret_tensor(buf5, (400, 512), (512, 1), 0); del buf5  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf19, (400, 512), (512, 1), 0), permute_109, out=buf20)
    del permute_109
    buf21 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf19, (512, 400), (1, 512), 0), view_161, out=buf21)
    del view_161
    buf22 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf23 = empty_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_sum_3(c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(slice_101.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf24 = aten.convolution_backward(buf23, getitem_85, primals_135, [192], [1, 1], [3, 3], [1, 1], False, [0, 0], 192, [True, True, True])
    del getitem_85
    buf25 = buf24[0]
    buf26 = buf24[1]
    buf27 = buf24[2]
    del buf24
    buf28 = buf23; del buf23  # reuse
    cpp_fused_convolution_backward_4(c_void_p(buf20.data_ptr()), c_void_p(slice_101.data_ptr()), c_void_p(buf28.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf29 = aten.convolution_backward(buf28, getitem_84, primals_133, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 192, [True, True, True])
    del buf28
    del getitem_84
    buf30 = buf29[0]
    buf31 = buf29[1]
    buf32 = buf29[2]
    del buf29
    buf33 = empty_strided((8, 128, 7, 7), (6272, 1, 896, 128), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_5(c_void_p(buf20.data_ptr()), c_void_p(slice_101.data_ptr()), c_void_p(buf33.data_ptr()))
    del slice_101
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf34 = aten.convolution_backward(buf33, getitem_83, primals_131, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, True])
    del buf33
    del getitem_83
    buf35 = buf34[0]
    buf36 = buf34[1]
    buf37 = buf34[2]
    del buf34
    buf38 = empty((8, 8, 50, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_6(c_void_p(buf20.data_ptr()), c_void_p(buf38.data_ptr()))
    buf39 = empty((64, 64, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_116, reinterpret_tensor(buf38, (64, 50, 64), (3200, 64, 1), 0), out=buf39)
    del permute_116
    buf40 = empty((64, 50, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf38, (64, 50, 64), (3200, 64, 1), 0), permute_117, out=buf40)
    del permute_117
    buf41 = reinterpret_tensor(buf38, (64, 50, 64), (3200, 64, 1), 0); del buf38  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_118, reinterpret_tensor(buf39, (64, 64, 64), (4096, 64, 1), 0), out=buf41)
    del permute_118
    buf42 = empty((64, 64, 50), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf39, (64, 64, 64), (4096, 64, 1), 0), permute_119, out=buf42)
    del permute_119
    buf43 = reinterpret_tensor(buf0, (8, 8, 1, 64), (512, 64, 4096, 1), 0); del buf0  # reuse
    buf44 = empty((24, 8, 50, 64), device='cpu', dtype=torch.float32)
    buf45 = empty((400, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_stack_view_7(c_void_p(buf42.data_ptr()), c_void_p(alias_8.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(cat_19.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()))
    del alias_8
    del buf25
    del cat_19
    buf46 = reinterpret_tensor(buf42, (400, 512), (512, 1), 0); del buf42  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf45, permute_122, out=buf46)
    del permute_122
    buf47 = empty((1536, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf45, (1536, 400), (1, 1536), 0), view_149, out=buf47)
    del view_149
    buf48 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf49 = buf16; del buf16  # reuse
    buf50 = buf15; del buf15  # reuse
    buf51 = empty((512, ), device='cpu', dtype=torch.float32)
    buf52 = empty((512, ), device='cpu', dtype=torch.float32)
    buf53 = buf19; del buf19  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_8(c_void_p(buf53.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(cat_18.data_ptr()), c_void_p(getitem_79.data_ptr()), c_void_p(rsqrt_18.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    del cat_18
    del getitem_79
    del primals_33
    del rsqrt_18
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf54 = aten.convolution_backward(reinterpret_tensor(buf53, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), view_147, primals_127, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 512, [True, True, True])
    del view_147
    buf55 = buf54[0]
    buf56 = buf54[1]
    buf57 = buf54[2]
    del buf54
    buf58 = reinterpret_tensor(buf46, (8, 50, 512), (25600, 512, 1), 0); del buf46  # reuse
    cpp_fused_add_select_backward_slice_backward_9(c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf58.data_ptr()))
    del buf55
    buf59 = reinterpret_tensor(buf11, (400, 2048), (2048, 1), 0); del buf11  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf58, (400, 512), (512, 1), 0), permute_128, out=buf59)
    del permute_128
    buf60 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf58, (512, 400), (1, 512), 0), view_145, out=buf60)
    del view_145
    buf61 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf62 = reinterpret_tensor(buf59, (8, 50, 2048), (102400, 2048, 1), 0); del buf59  # reuse
    cpp_fused_gelu_gelu_backward_sum_10(c_void_p(buf62.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(addmm_26.data_ptr()), c_void_p(buf61.data_ptr()))
    del addmm_26
    buf63 = reinterpret_tensor(buf53, (400, 512), (512, 1), 0); del buf53  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (400, 2048), (2048, 1), 0), permute_132, out=buf63)
    del permute_132
    buf64 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (2048, 400), (1, 2048), 0), view_143, out=buf64)
    del view_143
    buf65 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf66 = buf50; del buf50  # reuse
    buf67 = buf49; del buf49  # reuse
    buf68 = empty((512, ), device='cpu', dtype=torch.float32)
    buf69 = empty((512, ), device='cpu', dtype=torch.float32)
    buf70 = buf58; del buf58  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_11(c_void_p(buf70.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(mul_66.data_ptr()), c_void_p(div_11.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()))
    del buf62
    del div_11
    del mul_66
    del primals_31
    buf71 = buf63; del buf63  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (400, 512), (512, 1), 0), permute_136, out=buf71)
    del permute_136
    buf72 = reinterpret_tensor(buf39, (512, 512), (512, 1), 0); del buf39  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (512, 400), (1, 512), 0), view_141, out=buf72)
    del view_141
    buf73 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf74 = buf30; del buf30  # reuse
    cpp_fused_convolution_backward_sum_12(c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(slice_89.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf75 = aten.convolution_backward(buf74, getitem_75, primals_135, [192], [1, 1], [3, 3], [1, 1], False, [0, 0], 192, [True, True, True])
    del getitem_75
    del primals_135
    buf76 = buf75[0]
    buf77 = buf75[1]
    buf78 = buf75[2]
    del buf75
    buf79 = reinterpret_tensor(buf26, (192, 1, 7, 7), (49, 49, 7, 1), 0); del buf26  # reuse
    buf80 = buf27; del buf27  # reuse
    buf81 = buf74; del buf74  # reuse
    cpp_fused_add_convolution_backward_13(c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(slice_89.data_ptr()), c_void_p(buf81.data_ptr()))
    del buf77
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf82 = aten.convolution_backward(buf81, getitem_74, primals_133, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 192, [True, True, True])
    del buf81
    del getitem_74
    del primals_133
    buf83 = buf82[0]
    buf84 = buf82[1]
    buf85 = buf82[2]
    del buf82
    buf86 = reinterpret_tensor(buf31, (192, 1, 5, 5), (25, 25, 5, 1), 0); del buf31  # reuse
    buf87 = buf32; del buf32  # reuse
    buf88 = buf35; del buf35  # reuse
    cpp_fused_add_convolution_backward_14(c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(slice_89.data_ptr()), c_void_p(buf88.data_ptr()))
    del buf84
    del slice_89
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf89 = aten.convolution_backward(buf88, getitem_73, primals_131, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, True])
    del buf88
    del getitem_73
    del primals_131
    buf90 = buf89[0]
    buf91 = buf89[1]
    buf92 = buf89[2]
    del buf89
    buf93 = reinterpret_tensor(buf36, (128, 1, 3, 3), (9, 9, 3, 1), 0); del buf36  # reuse
    buf94 = buf37; del buf37  # reuse
    buf95 = reinterpret_tensor(buf41, (8, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf41  # reuse
    cpp_fused_add_clone_mul_15(c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf95.data_ptr()))
    del buf91
    buf96 = empty((64, 64, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_143, reinterpret_tensor(buf95, (64, 50, 64), (3200, 64, 1), 0), out=buf96)
    del permute_143
    buf97 = buf40; del buf40  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf95, (64, 50, 64), (3200, 64, 1), 0), permute_144, out=buf97)
    del permute_144
    buf98 = reinterpret_tensor(buf95, (64, 50, 64), (3200, 64, 1), 0); del buf95  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_145, reinterpret_tensor(buf96, (64, 64, 64), (4096, 64, 1), 0), out=buf98)
    del permute_145
    buf99 = reinterpret_tensor(buf20, (64, 64, 50), (3200, 50, 1), 0); del buf20  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf96, (64, 64, 64), (4096, 64, 1), 0), permute_146, out=buf99)
    del buf96
    del permute_146
    buf100 = buf43; del buf43  # reuse
    buf101 = reinterpret_tensor(buf45, (24, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf45  # reuse
    buf102 = reinterpret_tensor(buf44, (400, 1536), (1536, 1), 0); del buf44  # reuse
    cpp_fused__softmax_backward_data_stack_view_16(c_void_p(buf99.data_ptr()), c_void_p(alias_9.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(cat_17.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    del alias_9
    del buf101
    del buf71
    del buf76
    del buf83
    del buf90
    del buf97
    del buf98
    del cat_17
    buf103 = reinterpret_tensor(buf99, (400, 512), (512, 1), 0); del buf99  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf102, permute_149, out=buf103)
    del permute_149
    buf104 = empty((1536, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (1536, 400), (1, 1536), 0), view_129, out=buf104)
    del view_129
    buf105 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf106 = buf67; del buf67  # reuse
    buf107 = buf66; del buf66  # reuse
    buf108 = empty((512, ), device='cpu', dtype=torch.float32)
    buf109 = empty((512, ), device='cpu', dtype=torch.float32)
    buf110 = reinterpret_tensor(buf103, (8, 50, 512), (25600, 512, 1), 0); del buf103  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_17(c_void_p(buf110.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(cat_16.data_ptr()), c_void_p(getitem_69.data_ptr()), c_void_p(rsqrt_16.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del buf102
    del buf106
    del buf107
    del buf70
    del cat_16
    del getitem_69
    del primals_29
    del rsqrt_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf111 = aten.convolution_backward(reinterpret_tensor(buf110, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), view_127, primals_127, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 512, [True, True, True])
    del primals_127
    del view_127
    buf112 = buf111[0]
    buf113 = buf111[1]
    buf114 = buf111[2]
    del buf111
    buf115 = reinterpret_tensor(buf113, (512, 1, 3, 3), (9, 9, 3, 1), 0); del buf113  # reuse
    buf116 = buf114; del buf114  # reuse
    buf117 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf118 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf119 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf120 = empty((512, ), device='cpu', dtype=torch.float32)
    buf121 = empty((512, ), device='cpu', dtype=torch.float32)
    buf122 = buf112; del buf112  # reuse
    cpp_fused_add_convolution_backward_native_layer_norm_backward_sum_18(c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(mul_60.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()))
    del buf110
    del buf118
    del buf119
    del buf56
    del div_13
    del mul_60
    del primals_125
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf123 = aten.convolution_backward(buf122, clone_47, primals_123, [512], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del clone_47
    del primals_123
    buf124 = buf123[0]
    buf125 = buf123[1]
    buf126 = buf123[2]
    del buf123
    buf127 = empty((1576, 320), device='cpu', dtype=torch.float32)
    cpp_fused_slice_backward_view_19(c_void_p(buf124.data_ptr()), c_void_p(buf127.data_ptr()))
    buf128 = empty((1576, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf127, permute_157, out=buf128)
    del permute_157
    buf129 = empty((320, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf127, (320, 1576), (1, 320), 0), view_123, out=buf129)
    del view_123
    buf130 = empty((1, 320), device='cpu', dtype=torch.float32)
    buf131 = reinterpret_tensor(buf128, (8, 197, 1280), (252160, 1280, 1), 0); del buf128  # reuse
    cpp_fused_gelu_gelu_backward_sum_20(c_void_p(buf131.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf130.data_ptr()))
    del addmm_22
    buf132 = buf127; del buf127  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf131, (1576, 1280), (1280, 1), 0), permute_161, out=buf132)
    del permute_161
    buf133 = empty((1280, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf131, (1280, 1576), (1, 1280), 0), view_121, out=buf133)
    del view_121
    buf134 = empty((1, 1280), device='cpu', dtype=torch.float32)
    buf135 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf136 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf137 = empty((320, ), device='cpu', dtype=torch.float32)
    buf138 = empty((320, ), device='cpu', dtype=torch.float32)
    buf139 = reinterpret_tensor(buf132, (8, 197, 320), (63040, 320, 1), 0); del buf132  # reuse
    cpp_fused_add_native_layer_norm_backward_slice_backward_sum_21(c_void_p(buf139.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(mul_55.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()))
    del buf124
    del div_14
    del mul_55
    del primals_26
    buf140 = empty((1576, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (1576, 320), (320, 1), 0), permute_165, out=buf140)
    del permute_165
    buf141 = empty((320, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (320, 1576), (1, 320), 0), view_119, out=buf141)
    del view_119
    buf142 = empty((1, 320), device='cpu', dtype=torch.float32)
    buf143 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_sum_22(c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(slice_74.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf144 = aten.convolution_backward(buf143, getitem_63, primals_107, [120], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, True])
    del getitem_63
    buf145 = buf144[0]
    buf146 = buf144[1]
    buf147 = buf144[2]
    del buf144
    buf148 = buf143; del buf143  # reuse
    cpp_fused_convolution_backward_23(c_void_p(buf140.data_ptr()), c_void_p(slice_74.data_ptr()), c_void_p(buf148.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf149 = aten.convolution_backward(buf148, getitem_62, primals_105, [120], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, True])
    del buf148
    del getitem_62
    buf150 = buf149[0]
    buf151 = buf149[1]
    buf152 = buf149[2]
    del buf149
    buf153 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_24(c_void_p(buf140.data_ptr()), c_void_p(slice_74.data_ptr()), c_void_p(buf153.data_ptr()))
    del slice_74
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf154 = aten.convolution_backward(buf153, getitem_61, primals_103, [80], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, True])
    del buf153
    del getitem_61
    buf155 = buf154[0]
    buf156 = buf154[1]
    buf157 = buf154[2]
    del buf154
    buf158 = empty((8, 8, 197, 40), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_25(c_void_p(buf140.data_ptr()), c_void_p(buf158.data_ptr()))
    buf159 = empty((64, 40, 40), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_172, reinterpret_tensor(buf158, (64, 197, 40), (7880, 40, 1), 0), out=buf159)
    del permute_172
    buf160 = empty((64, 197, 40), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf158, (64, 197, 40), (7880, 40, 1), 0), permute_173, out=buf160)
    del permute_173
    buf161 = reinterpret_tensor(buf158, (64, 197, 40), (7880, 40, 1), 0); del buf158  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_174, reinterpret_tensor(buf159, (64, 40, 40), (1600, 40, 1), 0), out=buf161)
    del permute_174
    buf162 = empty((64, 40, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf159, (64, 40, 40), (1600, 40, 1), 0), permute_175, out=buf162)
    del permute_175
    buf163 = empty_strided((8, 8, 1, 40), (320, 40, 2560, 1), device='cpu', dtype=torch.float32)
    buf164 = empty((24, 8, 197, 40), device='cpu', dtype=torch.float32)
    buf165 = empty((1576, 960), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_stack_view_26(c_void_p(buf162.data_ptr()), c_void_p(alias_10.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(cat_14.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    del alias_10
    del buf145
    del cat_14
    buf166 = reinterpret_tensor(buf162, (1576, 320), (320, 1), 0); del buf162  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf165, permute_178, out=buf166)
    del permute_178
    buf167 = empty((960, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf165, (960, 1576), (1, 960), 0), view_107, out=buf167)
    del view_107
    buf168 = empty((1, 960), device='cpu', dtype=torch.float32)
    buf169 = buf136; del buf136  # reuse
    buf170 = buf135; del buf135  # reuse
    buf171 = empty((320, ), device='cpu', dtype=torch.float32)
    buf172 = empty((320, ), device='cpu', dtype=torch.float32)
    buf173 = buf139; del buf139  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_27(c_void_p(buf173.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(cat_13.data_ptr()), c_void_p(getitem_57.data_ptr()), c_void_p(rsqrt_13.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del cat_13
    del getitem_57
    del primals_24
    del rsqrt_13
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf174 = aten.convolution_backward(reinterpret_tensor(buf173, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), view_105, primals_99, [320], [1, 1], [1, 1], [1, 1], False, [0, 0], 320, [True, True, True])
    del view_105
    buf175 = buf174[0]
    buf176 = buf174[1]
    buf177 = buf174[2]
    del buf174
    buf178 = reinterpret_tensor(buf166, (8, 197, 320), (63040, 320, 1), 0); del buf166  # reuse
    cpp_fused_add_slice_backward_28(c_void_p(buf173.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf178.data_ptr()))
    del buf175
    buf179 = reinterpret_tensor(buf131, (1576, 1280), (1280, 1), 0); del buf131  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf178, (1576, 320), (320, 1), 0), permute_184, out=buf179)
    del permute_184
    buf180 = empty((320, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf178, (320, 1576), (1, 320), 0), view_103, out=buf180)
    del view_103
    buf181 = empty((1, 320), device='cpu', dtype=torch.float32)
    buf182 = reinterpret_tensor(buf179, (8, 197, 1280), (252160, 1280, 1), 0); del buf179  # reuse
    cpp_fused_gelu_gelu_backward_sum_29(c_void_p(buf182.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(addmm_18.data_ptr()), c_void_p(buf181.data_ptr()))
    del addmm_18
    buf183 = reinterpret_tensor(buf173, (1576, 320), (320, 1), 0); del buf173  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (1576, 1280), (1280, 1), 0), permute_188, out=buf183)
    del permute_188
    buf184 = empty((1280, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (1280, 1576), (1, 1280), 0), view_101, out=buf184)
    del view_101
    buf185 = empty((1, 1280), device='cpu', dtype=torch.float32)
    buf186 = buf170; del buf170  # reuse
    buf187 = buf169; del buf169  # reuse
    buf188 = empty((320, ), device='cpu', dtype=torch.float32)
    buf189 = empty((320, ), device='cpu', dtype=torch.float32)
    buf190 = buf178; del buf178  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_30(c_void_p(buf190.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_46.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    del buf182
    del div_16
    del mul_46
    del primals_22
    buf191 = buf183; del buf183  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (1576, 320), (320, 1), 0), permute_192, out=buf191)
    del permute_192
    buf192 = reinterpret_tensor(buf159, (320, 320), (320, 1), 0); del buf159  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (320, 1576), (1, 320), 0), view_99, out=buf192)
    del view_99
    buf193 = empty((1, 320), device='cpu', dtype=torch.float32)
    buf194 = buf150; del buf150  # reuse
    cpp_fused_convolution_backward_sum_31(c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(slice_62.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf195 = aten.convolution_backward(buf194, getitem_53, primals_107, [120], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, True])
    del getitem_53
    del primals_107
    buf196 = buf195[0]
    buf197 = buf195[1]
    buf198 = buf195[2]
    del buf195
    buf199 = reinterpret_tensor(buf146, (120, 1, 7, 7), (49, 49, 7, 1), 0); del buf146  # reuse
    buf200 = buf147; del buf147  # reuse
    buf201 = buf194; del buf194  # reuse
    cpp_fused_add_convolution_backward_32(c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(slice_62.data_ptr()), c_void_p(buf201.data_ptr()))
    del buf197
    del buf198
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf202 = aten.convolution_backward(buf201, getitem_52, primals_105, [120], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, True])
    del buf201
    del getitem_52
    del primals_105
    buf203 = buf202[0]
    buf204 = buf202[1]
    buf205 = buf202[2]
    del buf202
    buf206 = reinterpret_tensor(buf151, (120, 1, 5, 5), (25, 25, 5, 1), 0); del buf151  # reuse
    buf207 = buf152; del buf152  # reuse
    buf208 = buf155; del buf155  # reuse
    cpp_fused_add_convolution_backward_33(c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(slice_62.data_ptr()), c_void_p(buf208.data_ptr()))
    del buf204
    del buf205
    del slice_62
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf209 = aten.convolution_backward(buf208, getitem_51, primals_103, [80], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, True])
    del buf208
    del getitem_51
    del primals_103
    buf210 = buf209[0]
    buf211 = buf209[1]
    buf212 = buf209[2]
    del buf209
    buf213 = reinterpret_tensor(buf156, (80, 1, 3, 3), (9, 9, 3, 1), 0); del buf156  # reuse
    buf214 = buf157; del buf157  # reuse
    buf215 = reinterpret_tensor(buf161, (8, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf161  # reuse
    cpp_fused_add_clone_mul_34(c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf215.data_ptr()))
    del buf211
    del buf212
    buf216 = empty((64, 40, 40), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_199, reinterpret_tensor(buf215, (64, 197, 40), (7880, 40, 1), 0), out=buf216)
    del permute_199
    buf217 = buf160; del buf160  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf215, (64, 197, 40), (7880, 40, 1), 0), permute_200, out=buf217)
    del permute_200
    buf218 = reinterpret_tensor(buf215, (64, 197, 40), (7880, 40, 1), 0); del buf215  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_201, reinterpret_tensor(buf216, (64, 40, 40), (1600, 40, 1), 0), out=buf218)
    del permute_201
    buf219 = reinterpret_tensor(buf140, (64, 40, 197), (7880, 197, 1), 0); del buf140  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf216, (64, 40, 40), (1600, 40, 1), 0), permute_202, out=buf219)
    del buf216
    del permute_202
    buf220 = buf163; del buf163  # reuse
    buf221 = reinterpret_tensor(buf165, (24, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf165  # reuse
    buf222 = reinterpret_tensor(buf164, (1576, 960), (960, 1), 0); del buf164  # reuse
    cpp_fused__softmax_backward_data_stack_view_35(c_void_p(buf219.data_ptr()), c_void_p(alias_11.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(cat_12.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()))
    del alias_11
    del buf191
    del buf196
    del buf203
    del buf210
    del buf217
    del buf218
    del buf220
    del buf221
    del cat_12
    buf223 = reinterpret_tensor(buf219, (1576, 320), (320, 1), 0); del buf219  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf222, permute_205, out=buf223)
    del permute_205
    buf224 = empty((960, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (960, 1576), (1, 960), 0), view_87, out=buf224)
    del view_87
    buf225 = empty((1, 960), device='cpu', dtype=torch.float32)
    buf226 = buf187; del buf187  # reuse
    buf227 = buf186; del buf186  # reuse
    buf228 = empty((320, ), device='cpu', dtype=torch.float32)
    buf229 = empty((320, ), device='cpu', dtype=torch.float32)
    buf230 = buf190; del buf190  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_36(c_void_p(buf230.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(cat_11.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(rsqrt_11.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()))
    del buf222
    del buf223
    del buf226
    del buf227
    del cat_11
    del getitem_47
    del primals_20
    del rsqrt_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf231 = aten.convolution_backward(reinterpret_tensor(buf230, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), view_85, primals_99, [320], [1, 1], [1, 1], [1, 1], False, [0, 0], 320, [True, True, True])
    del primals_99
    del view_85
    buf232 = buf231[0]
    buf233 = buf231[1]
    buf234 = buf231[2]
    del buf231
    buf235 = reinterpret_tensor(buf176, (320, 1, 3, 3), (9, 9, 3, 1), 0); del buf176  # reuse
    buf236 = buf177; del buf177  # reuse
    buf237 = empty((1, 1, 320), device='cpu', dtype=torch.float32)
    buf238 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf239 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf240 = empty((320, ), device='cpu', dtype=torch.float32)
    buf241 = empty((320, ), device='cpu', dtype=torch.float32)
    buf242 = buf232; del buf232  # reuse
    cpp_fused_add_convolution_backward_native_layer_norm_backward_sum_37(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(mul_40.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del buf230
    del buf233
    del buf234
    del buf238
    del buf239
    del div_18
    del mul_40
    del primals_97
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf243 = aten.convolution_backward(buf242, clone_31, primals_95, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf242
    del clone_31
    del primals_95
    buf244 = buf243[0]
    buf245 = buf243[1]
    buf246 = buf243[2]
    del buf243
    buf247 = empty((6280, 128), device='cpu', dtype=torch.float32)
    cpp_fused_slice_backward_view_38(c_void_p(buf244.data_ptr()), c_void_p(buf247.data_ptr()))
    buf248 = empty((6280, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf247, permute_213, out=buf248)
    del permute_213
    buf249 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf247, (128, 6280), (1, 128), 0), view_81, out=buf249)
    del view_81
    buf250 = reinterpret_tensor(buf92, (1, 128), (128, 1), 0); del buf92  # reuse
    buf251 = reinterpret_tensor(buf248, (8, 785, 1024), (803840, 1024, 1), 0); del buf248  # reuse
    cpp_fused_gelu_gelu_backward_sum_39(c_void_p(buf251.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(buf250.data_ptr()))
    del addmm_14
    buf252 = buf247; del buf247  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (6280, 1024), (1024, 1), 0), permute_217, out=buf252)
    del permute_217
    buf253 = empty((1024, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (1024, 6280), (1, 1024), 0), view_79, out=buf253)
    del view_79
    buf254 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf255 = empty_strided((8, 785, 1), (785, 1, 6280), device='cpu', dtype=torch.float32)
    buf256 = empty_strided((8, 785, 1), (785, 1, 6280), device='cpu', dtype=torch.float32)
    buf257 = empty((128, ), device='cpu', dtype=torch.float32)
    buf258 = empty((128, ), device='cpu', dtype=torch.float32)
    buf259 = reinterpret_tensor(buf252, (8, 785, 128), (100480, 128, 1), 0); del buf252  # reuse
    cpp_fused_add_native_layer_norm_backward_slice_backward_sum_40(c_void_p(buf259.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    del buf244
    del div_19
    del mul_35
    del primals_17
    buf260 = empty((6280, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf259, (6280, 128), (128, 1), 0), permute_221, out=buf260)
    del permute_221
    buf261 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf259, (128, 6280), (1, 128), 0), view_77, out=buf261)
    del view_77
    buf262 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf263 = empty_strided((8, 48, 28, 28), (37632, 1, 1344, 48), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_sum_41(c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(slice_47.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf264 = aten.convolution_backward(buf263, getitem_41, primals_79, [48], [1, 1], [3, 3], [1, 1], False, [0, 0], 48, [True, True, True])
    del getitem_41
    buf265 = buf264[0]
    buf266 = buf264[1]
    buf267 = buf264[2]
    del buf264
    buf268 = buf263; del buf263  # reuse
    cpp_fused_convolution_backward_42(c_void_p(buf260.data_ptr()), c_void_p(slice_47.data_ptr()), c_void_p(buf268.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf269 = aten.convolution_backward(buf268, getitem_40, primals_77, [48], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, True])
    del buf268
    del getitem_40
    buf270 = buf269[0]
    buf271 = buf269[1]
    buf272 = buf269[2]
    del buf269
    buf273 = reinterpret_tensor(buf122, (8, 32, 28, 28), (25088, 1, 896, 32), 0); del buf122  # reuse
    cpp_fused_convolution_backward_43(c_void_p(buf260.data_ptr()), c_void_p(slice_47.data_ptr()), c_void_p(buf273.data_ptr()))
    del slice_47
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf274 = aten.convolution_backward(buf273, getitem_39, primals_75, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, True])
    del buf273
    del getitem_39
    buf275 = buf274[0]
    buf276 = buf274[1]
    buf277 = buf274[2]
    del buf274
    buf278 = empty((8, 8, 785, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_44(c_void_p(buf260.data_ptr()), c_void_p(buf278.data_ptr()))
    buf279 = empty((64, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_228, reinterpret_tensor(buf278, (64, 785, 16), (12560, 16, 1), 0), out=buf279)
    del permute_228
    buf280 = empty((64, 785, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf278, (64, 785, 16), (12560, 16, 1), 0), permute_229, out=buf280)
    del permute_229
    buf281 = reinterpret_tensor(buf278, (64, 785, 16), (12560, 16, 1), 0); del buf278  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_230, reinterpret_tensor(buf279, (64, 16, 16), (256, 16, 1), 0), out=buf281)
    del permute_230
    buf282 = empty((64, 16, 785), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf279, (64, 16, 16), (256, 16, 1), 0), permute_231, out=buf282)
    del permute_231
    buf283 = empty_strided((8, 8, 1, 16), (128, 16, 1024, 1), device='cpu', dtype=torch.float32)
    buf284 = empty((24, 8, 785, 16), device='cpu', dtype=torch.float32)
    buf285 = empty((6280, 384), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_stack_view_45(c_void_p(buf282.data_ptr()), c_void_p(alias_12.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(cat_9.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    del alias_12
    del buf265
    del cat_9
    buf286 = reinterpret_tensor(buf282, (6280, 128), (128, 1), 0); del buf282  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf285, permute_234, out=buf286)
    del permute_234
    buf287 = empty((384, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf285, (384, 6280), (1, 384), 0), view_65, out=buf287)
    del view_65
    buf288 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf289 = buf256; del buf256  # reuse
    buf290 = buf255; del buf255  # reuse
    buf291 = empty((128, ), device='cpu', dtype=torch.float32)
    buf292 = empty((128, ), device='cpu', dtype=torch.float32)
    buf293 = buf259; del buf259  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_46(c_void_p(buf293.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(cat_8.data_ptr()), c_void_p(getitem_35.data_ptr()), c_void_p(rsqrt_8.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()))
    del cat_8
    del getitem_35
    del primals_15
    del rsqrt_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf294 = aten.convolution_backward(reinterpret_tensor(buf293, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), view_63, primals_71, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, True])
    del view_63
    buf295 = buf294[0]
    buf296 = buf294[1]
    buf297 = buf294[2]
    del buf294
    buf298 = reinterpret_tensor(buf286, (8, 785, 128), (100480, 128, 1), 0); del buf286  # reuse
    cpp_fused_add_slice_backward_47(c_void_p(buf293.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf298.data_ptr()))
    del buf295
    buf299 = reinterpret_tensor(buf251, (6280, 1024), (1024, 1), 0); del buf251  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf298, (6280, 128), (128, 1), 0), permute_240, out=buf299)
    del permute_240
    buf300 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf298, (128, 6280), (1, 128), 0), view_61, out=buf300)
    del view_61
    buf301 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf302 = reinterpret_tensor(buf299, (8, 785, 1024), (803840, 1024, 1), 0); del buf299  # reuse
    cpp_fused_gelu_gelu_backward_sum_48(c_void_p(buf302.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf301.data_ptr()))
    del addmm_10
    buf303 = reinterpret_tensor(buf293, (6280, 128), (128, 1), 0); del buf293  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (6280, 1024), (1024, 1), 0), permute_244, out=buf303)
    del permute_244
    buf304 = empty((1024, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (1024, 6280), (1, 1024), 0), view_59, out=buf304)
    del view_59
    buf305 = reinterpret_tensor(buf283, (1, 1024), (1024, 1), 0); del buf283  # reuse
    buf306 = buf290; del buf290  # reuse
    buf307 = buf289; del buf289  # reuse
    buf308 = empty((128, ), device='cpu', dtype=torch.float32)
    buf309 = empty((128, ), device='cpu', dtype=torch.float32)
    buf310 = buf298; del buf298  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_49(c_void_p(buf310.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(mul_26.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()))
    del buf302
    del div_21
    del mul_26
    del primals_13
    buf311 = buf303; del buf303  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf310, (6280, 128), (128, 1), 0), permute_248, out=buf311)
    del permute_248
    buf312 = reinterpret_tensor(buf279, (128, 128), (128, 1), 0); del buf279  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf310, (128, 6280), (1, 128), 0), view_57, out=buf312)
    del view_57
    buf313 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf314 = buf270; del buf270  # reuse
    cpp_fused_convolution_backward_sum_50(c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(slice_35.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf315 = aten.convolution_backward(buf314, getitem_31, primals_79, [48], [1, 1], [3, 3], [1, 1], False, [0, 0], 48, [True, True, True])
    del getitem_31
    del primals_79
    buf316 = buf315[0]
    buf317 = buf315[1]
    buf318 = buf315[2]
    del buf315
    buf319 = reinterpret_tensor(buf266, (48, 1, 7, 7), (49, 49, 7, 1), 0); del buf266  # reuse
    buf320 = buf267; del buf267  # reuse
    buf321 = buf314; del buf314  # reuse
    cpp_fused_add_convolution_backward_51(c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(slice_35.data_ptr()), c_void_p(buf321.data_ptr()))
    del buf317
    del buf318
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf322 = aten.convolution_backward(buf321, getitem_30, primals_77, [48], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, True])
    del buf321
    del getitem_30
    del primals_77
    buf323 = buf322[0]
    buf324 = buf322[1]
    buf325 = buf322[2]
    del buf322
    buf326 = reinterpret_tensor(buf271, (48, 1, 5, 5), (25, 25, 5, 1), 0); del buf271  # reuse
    buf327 = buf272; del buf272  # reuse
    buf328 = buf275; del buf275  # reuse
    cpp_fused_add_convolution_backward_52(c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(slice_35.data_ptr()), c_void_p(buf328.data_ptr()))
    del buf324
    del buf325
    del slice_35
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf329 = aten.convolution_backward(buf328, getitem_29, primals_75, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, True])
    del buf328
    del getitem_29
    del primals_75
    buf330 = buf329[0]
    buf331 = buf329[1]
    buf332 = buf329[2]
    del buf329
    buf333 = reinterpret_tensor(buf276, (32, 1, 3, 3), (9, 9, 3, 1), 0); del buf276  # reuse
    buf334 = buf277; del buf277  # reuse
    buf335 = reinterpret_tensor(buf281, (8, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf281  # reuse
    cpp_fused_add_clone_mul_53(c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf335.data_ptr()))
    del buf331
    del buf332
    buf336 = empty((64, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_255, reinterpret_tensor(buf335, (64, 785, 16), (12560, 16, 1), 0), out=buf336)
    del permute_255
    buf337 = buf280; del buf280  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf335, (64, 785, 16), (12560, 16, 1), 0), permute_256, out=buf337)
    del permute_256
    buf338 = reinterpret_tensor(buf335, (64, 785, 16), (12560, 16, 1), 0); del buf335  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_257, reinterpret_tensor(buf336, (64, 16, 16), (256, 16, 1), 0), out=buf338)
    del permute_257
    buf339 = reinterpret_tensor(buf260, (64, 16, 785), (12560, 785, 1), 0); del buf260  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf336, (64, 16, 16), (256, 16, 1), 0), permute_258, out=buf339)
    del buf336
    del permute_258
    buf340 = empty_strided((8, 8, 1, 16), (128, 16, 1024, 1), device='cpu', dtype=torch.float32)
    buf341 = reinterpret_tensor(buf285, (24, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf285  # reuse
    buf342 = reinterpret_tensor(buf284, (6280, 384), (384, 1), 0); del buf284  # reuse
    cpp_fused__softmax_backward_data_stack_view_54(c_void_p(buf339.data_ptr()), c_void_p(alias_13.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(cat_7.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    del alias_13
    del buf311
    del buf316
    del buf323
    del buf330
    del buf337
    del buf338
    del buf340
    del buf341
    del cat_7
    buf343 = reinterpret_tensor(buf339, (6280, 128), (128, 1), 0); del buf339  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf342, permute_261, out=buf343)
    del permute_261
    buf344 = empty((384, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf342, (384, 6280), (1, 384), 0), view_45, out=buf344)
    del view_45
    buf345 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf346 = buf307; del buf307  # reuse
    buf347 = buf306; del buf306  # reuse
    buf348 = empty((128, ), device='cpu', dtype=torch.float32)
    buf349 = empty((128, ), device='cpu', dtype=torch.float32)
    buf350 = buf310; del buf310  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_55(c_void_p(buf350.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(cat_6.data_ptr()), c_void_p(getitem_25.data_ptr()), c_void_p(rsqrt_6.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()))
    del buf342
    del buf343
    del buf346
    del buf347
    del cat_6
    del getitem_25
    del primals_11
    del rsqrt_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf351 = aten.convolution_backward(reinterpret_tensor(buf350, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), view_43, primals_71, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, True])
    del primals_71
    del view_43
    buf352 = buf351[0]
    buf353 = buf351[1]
    buf354 = buf351[2]
    del buf351
    buf355 = reinterpret_tensor(buf296, (128, 1, 3, 3), (9, 9, 3, 1), 0); del buf296  # reuse
    buf356 = buf297; del buf297  # reuse
    buf357 = empty((1, 1, 128), device='cpu', dtype=torch.float32)
    buf358 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf359 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf360 = empty((128, ), device='cpu', dtype=torch.float32)
    buf361 = empty((128, ), device='cpu', dtype=torch.float32)
    buf362 = buf352; del buf352  # reuse
    cpp_fused_add_convolution_backward_native_layer_norm_backward_sum_56(c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(mul_20.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()))
    del buf350
    del buf353
    del buf354
    del buf358
    del buf359
    del div_23
    del mul_20
    del primals_69
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf363 = aten.convolution_backward(buf362, clone_15, primals_67, [128], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf362
    del clone_15
    del primals_67
    buf364 = buf363[0]
    buf365 = buf363[1]
    buf366 = buf363[2]
    del buf363
    buf367 = empty((25096, 64), device='cpu', dtype=torch.float32)
    cpp_fused_slice_backward_view_57(c_void_p(buf364.data_ptr()), c_void_p(buf367.data_ptr()))
    buf368 = empty((25096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf367, permute_269, out=buf368)
    del permute_269
    buf369 = empty((64, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf367, (64, 25096), (1, 64), 0), view_39, out=buf369)
    del view_39
    buf370 = empty((1, 64), device='cpu', dtype=torch.float32)
    buf371 = reinterpret_tensor(buf368, (8, 3137, 512), (1606144, 512, 1), 0); del buf368  # reuse
    cpp_fused_gelu_gelu_backward_sum_58(c_void_p(buf371.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(buf370.data_ptr()))
    del addmm_6
    buf372 = buf367; del buf367  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (25096, 512), (512, 1), 0), permute_273, out=buf372)
    del permute_273
    buf373 = empty((512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (512, 25096), (1, 512), 0), view_37, out=buf373)
    del view_37
    buf374 = reinterpret_tensor(buf57, (1, 512), (512, 1), 0); del buf57  # reuse
    buf375 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cpu', dtype=torch.float32)
    buf376 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cpu', dtype=torch.float32)
    buf377 = empty((64, ), device='cpu', dtype=torch.float32)
    buf378 = empty((64, ), device='cpu', dtype=torch.float32)
    buf379 = reinterpret_tensor(buf372, (8, 3137, 64), (200768, 64, 1), 0); del buf372  # reuse
    cpp_fused_add_native_layer_norm_backward_slice_backward_sum_59(c_void_p(buf379.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(mul_15.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()))
    del buf364
    del div_24
    del mul_15
    del primals_8
    buf380 = empty((25096, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf379, (25096, 64), (64, 1), 0), permute_277, out=buf380)
    del permute_277
    buf381 = reinterpret_tensor(buf100, (64, 64), (64, 1), 0); del buf100  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf379, (64, 25096), (1, 64), 0), view_35, out=buf381)
    del view_35
    buf382 = empty((1, 64), device='cpu', dtype=torch.float32)
    buf383 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_sum_60(c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(slice_20.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf384 = aten.convolution_backward(buf383, getitem_19, primals_51, [24], [1, 1], [3, 3], [1, 1], False, [0, 0], 24, [True, True, True])
    del getitem_19
    buf385 = buf384[0]
    buf386 = buf384[1]
    buf387 = buf384[2]
    del buf384
    buf388 = buf383; del buf383  # reuse
    cpp_fused_convolution_backward_61(c_void_p(buf380.data_ptr()), c_void_p(slice_20.data_ptr()), c_void_p(buf388.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf389 = aten.convolution_backward(buf388, getitem_18, primals_49, [24], [1, 1], [2, 2], [1, 1], False, [0, 0], 24, [True, True, True])
    del buf388
    del getitem_18
    buf390 = buf389[0]
    buf391 = buf389[1]
    buf392 = buf389[2]
    del buf389
    buf393 = empty_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_62(c_void_p(buf380.data_ptr()), c_void_p(slice_20.data_ptr()), c_void_p(buf393.data_ptr()))
    del slice_20
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf394 = aten.convolution_backward(buf393, getitem_17, primals_47, [16], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, True])
    del buf393
    del getitem_17
    buf395 = buf394[0]
    buf396 = buf394[1]
    buf397 = buf394[2]
    del buf394
    buf398 = empty((8, 8, 3137, 8), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_63(c_void_p(buf380.data_ptr()), c_void_p(buf398.data_ptr()))
    buf399 = empty((64, 8, 8), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_284, reinterpret_tensor(buf398, (64, 3137, 8), (25096, 8, 1), 0), out=buf399)
    del permute_284
    buf400 = empty((64, 3137, 8), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf398, (64, 3137, 8), (25096, 8, 1), 0), permute_285, out=buf400)
    del permute_285
    buf401 = reinterpret_tensor(buf398, (64, 3137, 8), (25096, 8, 1), 0); del buf398  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_286, reinterpret_tensor(buf399, (64, 8, 8), (64, 8, 1), 0), out=buf401)
    del permute_286
    buf402 = empty((64, 8, 3137), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf399, (64, 8, 8), (64, 8, 1), 0), permute_287, out=buf402)
    del permute_287
    buf403 = empty_strided((8, 8, 1, 8), (64, 8, 512, 1), device='cpu', dtype=torch.float32)
    buf404 = empty((24, 8, 3137, 8), device='cpu', dtype=torch.float32)
    buf405 = empty((25096, 192), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_stack_view_64(c_void_p(buf402.data_ptr()), c_void_p(alias_14.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(cat_4.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()))
    del alias_14
    del buf385
    del cat_4
    buf406 = reinterpret_tensor(buf402, (25096, 64), (64, 1), 0); del buf402  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf405, permute_290, out=buf406)
    del permute_290
    buf407 = empty((192, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf405, (192, 25096), (1, 192), 0), view_23, out=buf407)
    del view_23
    buf408 = reinterpret_tensor(buf85, (1, 192), (192, 1), 0); del buf85  # reuse
    buf409 = buf376; del buf376  # reuse
    buf410 = buf375; del buf375  # reuse
    buf411 = empty((64, ), device='cpu', dtype=torch.float32)
    buf412 = empty((64, ), device='cpu', dtype=torch.float32)
    buf413 = buf379; del buf379  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_65(c_void_p(buf413.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(cat_3.data_ptr()), c_void_p(getitem_13.data_ptr()), c_void_p(rsqrt_3.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()))
    del cat_3
    del getitem_13
    del primals_6
    del rsqrt_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf414 = aten.convolution_backward(reinterpret_tensor(buf413, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), view_21, primals_43, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, True])
    del view_21
    buf415 = buf414[0]
    buf416 = buf414[1]
    buf417 = buf414[2]
    del buf414
    buf418 = reinterpret_tensor(buf406, (8, 3137, 64), (200768, 64, 1), 0); del buf406  # reuse
    cpp_fused_add_slice_backward_66(c_void_p(buf413.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf418.data_ptr()))
    del buf415
    buf419 = reinterpret_tensor(buf371, (25096, 512), (512, 1), 0); del buf371  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf418, (25096, 64), (64, 1), 0), permute_296, out=buf419)
    del permute_296
    buf420 = empty((64, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf418, (64, 25096), (1, 64), 0), view_19, out=buf420)
    del view_19
    buf421 = empty((1, 64), device='cpu', dtype=torch.float32)
    buf422 = reinterpret_tensor(buf419, (8, 3137, 512), (1606144, 512, 1), 0); del buf419  # reuse
    cpp_fused_gelu_gelu_backward_sum_67(c_void_p(buf422.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(buf421.data_ptr()))
    del addmm_2
    buf423 = reinterpret_tensor(buf413, (25096, 64), (64, 1), 0); del buf413  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (25096, 512), (512, 1), 0), permute_300, out=buf423)
    del permute_300
    buf424 = empty((512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (512, 25096), (1, 512), 0), view_17, out=buf424)
    del view_17
    buf425 = reinterpret_tensor(buf403, (1, 512), (512, 1), 0); del buf403  # reuse
    buf426 = buf410; del buf410  # reuse
    buf427 = buf409; del buf409  # reuse
    buf428 = empty((64, ), device='cpu', dtype=torch.float32)
    buf429 = empty((64, ), device='cpu', dtype=torch.float32)
    buf430 = buf418; del buf418  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_68(c_void_p(buf430.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(mul_6.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()))
    del buf422
    del div_26
    del mul_6
    del primals_4
    buf431 = buf423; del buf423  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf430, (25096, 64), (64, 1), 0), permute_304, out=buf431)
    del permute_304
    buf432 = reinterpret_tensor(buf399, (64, 64), (64, 1), 0); del buf399  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf430, (64, 25096), (1, 64), 0), view_15, out=buf432)
    del view_15
    buf433 = empty((1, 64), device='cpu', dtype=torch.float32)
    buf434 = buf390; del buf390  # reuse
    cpp_fused_convolution_backward_sum_69(c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(slice_8.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf435 = aten.convolution_backward(buf434, getitem_9, primals_51, [24], [1, 1], [3, 3], [1, 1], False, [0, 0], 24, [True, True, True])
    del getitem_9
    del primals_51
    buf436 = buf435[0]
    buf437 = buf435[1]
    buf438 = buf435[2]
    del buf435
    buf439 = reinterpret_tensor(buf386, (24, 1, 7, 7), (49, 49, 7, 1), 0); del buf386  # reuse
    buf440 = buf387; del buf387  # reuse
    buf441 = buf434; del buf434  # reuse
    cpp_fused_add_convolution_backward_70(c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(slice_8.data_ptr()), c_void_p(buf441.data_ptr()))
    del buf437
    del buf438
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf442 = aten.convolution_backward(buf441, getitem_8, primals_49, [24], [1, 1], [2, 2], [1, 1], False, [0, 0], 24, [True, True, True])
    del buf441
    del getitem_8
    del primals_49
    buf443 = buf442[0]
    buf444 = buf442[1]
    buf445 = buf442[2]
    del buf442
    buf446 = reinterpret_tensor(buf391, (24, 1, 5, 5), (25, 25, 5, 1), 0); del buf391  # reuse
    buf447 = buf392; del buf392  # reuse
    buf448 = buf395; del buf395  # reuse
    cpp_fused_add_convolution_backward_71(c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(slice_8.data_ptr()), c_void_p(buf448.data_ptr()))
    del buf444
    del buf445
    del slice_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf449 = aten.convolution_backward(buf448, getitem_7, primals_47, [16], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, True])
    del buf448
    del getitem_7
    del primals_47
    buf450 = buf449[0]
    buf451 = buf449[1]
    buf452 = buf449[2]
    del buf449
    buf453 = reinterpret_tensor(buf396, (16, 1, 3, 3), (9, 9, 3, 1), 0); del buf396  # reuse
    buf454 = buf397; del buf397  # reuse
    buf455 = reinterpret_tensor(buf401, (8, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf401  # reuse
    cpp_fused_add_clone_mul_72(c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf455.data_ptr()))
    del buf451
    del buf452
    buf456 = empty((64, 8, 8), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_311, reinterpret_tensor(buf455, (64, 3137, 8), (25096, 8, 1), 0), out=buf456)
    del permute_311
    buf457 = buf400; del buf400  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf455, (64, 3137, 8), (25096, 8, 1), 0), permute_312, out=buf457)
    del permute_312
    buf458 = reinterpret_tensor(buf455, (64, 3137, 8), (25096, 8, 1), 0); del buf455  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_313, reinterpret_tensor(buf456, (64, 8, 8), (64, 8, 1), 0), out=buf458)
    del permute_313
    buf459 = reinterpret_tensor(buf380, (64, 8, 3137), (25096, 3137, 1), 0); del buf380  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf456, (64, 8, 8), (64, 8, 1), 0), permute_314, out=buf459)
    del buf456
    del permute_314
    buf460 = empty_strided((8, 8, 1, 8), (64, 8, 512, 1), device='cpu', dtype=torch.float32)
    buf461 = reinterpret_tensor(buf405, (24, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf405  # reuse
    buf462 = reinterpret_tensor(buf404, (25096, 192), (192, 1), 0); del buf404  # reuse
    cpp_fused__softmax_backward_data_stack_view_73(c_void_p(buf459.data_ptr()), c_void_p(alias_15.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(cat_2.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()))
    del alias_15
    del buf431
    del buf436
    del buf443
    del buf450
    del buf457
    del buf458
    del buf460
    del buf461
    del cat_2
    buf463 = reinterpret_tensor(buf459, (25096, 64), (64, 1), 0); del buf459  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf462, permute_317, out=buf463)
    del permute_317
    buf464 = empty((192, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf462, (192, 25096), (1, 192), 0), view_3, out=buf464)
    del view_3
    buf465 = reinterpret_tensor(buf78, (1, 192), (192, 1), 0); del buf78  # reuse
    buf466 = buf427; del buf427  # reuse
    buf467 = buf426; del buf426  # reuse
    buf468 = empty((64, ), device='cpu', dtype=torch.float32)
    buf469 = empty((64, ), device='cpu', dtype=torch.float32)
    buf470 = buf430; del buf430  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_74(c_void_p(buf470.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(cat_1.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(rsqrt_1.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()))
    del buf462
    del buf463
    del buf466
    del buf467
    del cat_1
    del getitem_3
    del primals_2
    del rsqrt_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf471 = aten.convolution_backward(reinterpret_tensor(buf470, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), view_1, primals_43, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, True])
    del primals_43
    del view_1
    buf472 = buf471[0]
    buf473 = buf471[1]
    buf474 = buf471[2]
    del buf471
    buf475 = reinterpret_tensor(buf416, (64, 1, 3, 3), (9, 9, 3, 1), 0); del buf416  # reuse
    buf476 = buf417; del buf417  # reuse
    buf477 = empty((1, 1, 64), device='cpu', dtype=torch.float32)
    buf478 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf479 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf480 = empty((64, ), device='cpu', dtype=torch.float32)
    buf481 = empty((64, ), device='cpu', dtype=torch.float32)
    buf482 = buf472; del buf472  # reuse
    cpp_fused_add_convolution_backward_native_layer_norm_backward_sum_75(c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()))
    del buf470
    del buf473
    del buf474
    del buf478
    del buf479
    del div_28
    del mul
    del primals_41
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf483 = aten.convolution_backward(buf482, primals_153, primals_39, [64], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf482
    del primals_153
    del primals_39
    buf484 = buf483[1]
    buf485 = buf483[2]
    return (buf477, buf468, buf469, buf428, buf429, buf411, buf412, buf377, buf378, buf357, buf348, buf349, buf308, buf309, buf291, buf292, buf257, buf258, buf237, buf228, buf229, buf188, buf189, buf171, buf172, buf137, buf138, buf117, buf108, buf109, buf68, buf69, buf51, buf52, buf17, buf18, buf6, buf7, buf484, buf485, buf480, buf481, buf475, buf476, reinterpret_tensor(buf464, (192, 64), (64, 1), 0), reinterpret_tensor(buf465, (192, ), (1, ), 0), buf453, buf454, buf446, buf447, buf439, buf440, reinterpret_tensor(buf432, (64, 64), (64, 1), 0), reinterpret_tensor(buf433, (64, ), (1, ), 0), reinterpret_tensor(buf424, (512, 64), (64, 1), 0), reinterpret_tensor(buf425, (512, ), (1, ), 0), reinterpret_tensor(buf420, (64, 512), (512, 1), 0), reinterpret_tensor(buf421, (64, ), (1, ), 0), reinterpret_tensor(buf407, (192, 64), (64, 1), 0), reinterpret_tensor(buf408, (192, ), (1, ), 0), reinterpret_tensor(buf381, (64, 64), (64, 1), 0), reinterpret_tensor(buf382, (64, ), (1, ), 0), reinterpret_tensor(buf373, (512, 64), (64, 1), 0), reinterpret_tensor(buf374, (512, ), (1, ), 0), reinterpret_tensor(buf369, (64, 512), (512, 1), 0), reinterpret_tensor(buf370, (64, ), (1, ), 0), buf365, buf366, buf360, buf361, buf355, buf356, reinterpret_tensor(buf344, (384, 128), (128, 1), 0), reinterpret_tensor(buf345, (384, ), (1, ), 0), buf333, buf334, buf326, buf327, buf319, buf320, reinterpret_tensor(buf312, (128, 128), (128, 1), 0), reinterpret_tensor(buf313, (128, ), (1, ), 0), reinterpret_tensor(buf304, (1024, 128), (128, 1), 0), reinterpret_tensor(buf305, (1024, ), (1, ), 0), reinterpret_tensor(buf300, (128, 1024), (1024, 1), 0), reinterpret_tensor(buf301, (128, ), (1, ), 0), reinterpret_tensor(buf287, (384, 128), (128, 1), 0), reinterpret_tensor(buf288, (384, ), (1, ), 0), reinterpret_tensor(buf261, (128, 128), (128, 1), 0), reinterpret_tensor(buf262, (128, ), (1, ), 0), reinterpret_tensor(buf253, (1024, 128), (128, 1), 0), reinterpret_tensor(buf254, (1024, ), (1, ), 0), reinterpret_tensor(buf249, (128, 1024), (1024, 1), 0), reinterpret_tensor(buf250, (128, ), (1, ), 0), buf245, buf246, buf240, buf241, buf235, buf236, reinterpret_tensor(buf224, (960, 320), (320, 1), 0), reinterpret_tensor(buf225, (960, ), (1, ), 0), buf213, buf214, buf206, buf207, buf199, buf200, reinterpret_tensor(buf192, (320, 320), (320, 1), 0), reinterpret_tensor(buf193, (320, ), (1, ), 0), reinterpret_tensor(buf184, (1280, 320), (320, 1), 0), reinterpret_tensor(buf185, (1280, ), (1, ), 0), reinterpret_tensor(buf180, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf181, (320, ), (1, ), 0), reinterpret_tensor(buf167, (960, 320), (320, 1), 0), reinterpret_tensor(buf168, (960, ), (1, ), 0), reinterpret_tensor(buf141, (320, 320), (320, 1), 0), reinterpret_tensor(buf142, (320, ), (1, ), 0), reinterpret_tensor(buf133, (1280, 320), (320, 1), 0), reinterpret_tensor(buf134, (1280, ), (1, ), 0), reinterpret_tensor(buf129, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf130, (320, ), (1, ), 0), buf125, buf126, buf120, buf121, buf115, buf116, reinterpret_tensor(buf104, (1536, 512), (512, 1), 0), reinterpret_tensor(buf105, (1536, ), (1, ), 0), buf93, buf94, buf86, buf87, buf79, buf80, reinterpret_tensor(buf72, (512, 512), (512, 1), 0), reinterpret_tensor(buf73, (512, ), (1, ), 0), reinterpret_tensor(buf64, (2048, 512), (512, 1), 0), reinterpret_tensor(buf65, (2048, ), (1, ), 0), reinterpret_tensor(buf60, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf61, (512, ), (1, ), 0), reinterpret_tensor(buf47, (1536, 512), (512, 1), 0), reinterpret_tensor(buf48, (1536, ), (1, ), 0), reinterpret_tensor(buf21, (512, 512), (512, 1), 0), reinterpret_tensor(buf22, (512, ), (1, ), 0), reinterpret_tensor(buf13, (2048, 512), (512, 1), 0), reinterpret_tensor(buf14, (2048, ), (1, ), 0), reinterpret_tensor(buf9, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf10, (512, ), (1, ), 0), reinterpret_tensor(buf1, (1000, 512), (512, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((64, 3, 4, 4), (48, 1, 12, 3), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((24, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((48, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((320, 128, 2, 2), (512, 1, 256, 128), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((512, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((192, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    mul = rand_strided((8, 3136, 64), (200704, 64, 1), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((8, 64, 56, 56), (200768, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((8, 3137, 64), (200768, 64, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((8, 3137, 1), (3137, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_1 = rand_strided((8, 3137, 1), (3137, 1, 1), device='cpu', dtype=torch.float32)
    view_3 = rand_strided((25096, 64), (64, 1), device='cpu', dtype=torch.float32)
    slice_8 = rand_strided((8, 8, 3136, 8), (602304, 8, 192, 1), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((8, 16, 56, 56), (602304, 1, 10752, 192), device='cpu', dtype=torch.float32)
    getitem_8 = rand_strided((8, 24, 56, 56), (602304, 1, 10752, 192), device='cpu', dtype=torch.float32)
    getitem_9 = rand_strided((8, 24, 56, 56), (602304, 1, 10752, 192), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    view_15 = rand_strided((25096, 64), (64, 1), device='cpu', dtype=torch.float32)
    mul_6 = rand_strided((8, 3137, 64), (200768, 64, 1), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((25096, 64), (64, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((25096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_19 = rand_strided((25096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((8, 64, 56, 56), (200768, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cat_3 = rand_strided((8, 3137, 64), (200768, 64, 1), device='cpu', dtype=torch.float32)
    getitem_13 = rand_strided((8, 3137, 1), (3137, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_3 = rand_strided((8, 3137, 1), (3137, 1, 1), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((25096, 64), (64, 1), device='cpu', dtype=torch.float32)
    slice_20 = rand_strided((8, 8, 3136, 8), (602304, 8, 192, 1), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((8, 16, 56, 56), (602304, 1, 10752, 192), device='cpu', dtype=torch.float32)
    getitem_18 = rand_strided((8, 24, 56, 56), (602304, 1, 10752, 192), device='cpu', dtype=torch.float32)
    getitem_19 = rand_strided((8, 24, 56, 56), (602304, 1, 10752, 192), device='cpu', dtype=torch.float32)
    cat_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    view_35 = rand_strided((25096, 64), (64, 1), device='cpu', dtype=torch.float32)
    mul_15 = rand_strided((8, 3137, 64), (200768, 64, 1), device='cpu', dtype=torch.float32)
    view_37 = rand_strided((25096, 64), (64, 1), device='cpu', dtype=torch.float32)
    addmm_6 = rand_strided((25096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_39 = rand_strided((25096, 512), (512, 1), device='cpu', dtype=torch.float32)
    clone_15 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    mul_20 = rand_strided((8, 784, 128), (100352, 128, 1), device='cpu', dtype=torch.float32)
    view_43 = rand_strided((8, 128, 28, 28), (100480, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cat_6 = rand_strided((8, 785, 128), (100480, 128, 1), device='cpu', dtype=torch.float32)
    getitem_25 = rand_strided((8, 785, 1), (785, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_6 = rand_strided((8, 785, 1), (785, 1, 1), device='cpu', dtype=torch.float32)
    view_45 = rand_strided((6280, 128), (128, 1), device='cpu', dtype=torch.float32)
    slice_35 = rand_strided((8, 8, 784, 16), (301440, 16, 384, 1), device='cpu', dtype=torch.float32)
    getitem_29 = rand_strided((8, 32, 28, 28), (301440, 1, 10752, 384), device='cpu', dtype=torch.float32)
    getitem_30 = rand_strided((8, 48, 28, 28), (301440, 1, 10752, 384), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((8, 48, 28, 28), (301440, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cat_7 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    view_57 = rand_strided((6280, 128), (128, 1), device='cpu', dtype=torch.float32)
    mul_26 = rand_strided((8, 785, 128), (100480, 128, 1), device='cpu', dtype=torch.float32)
    view_59 = rand_strided((6280, 128), (128, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((6280, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_61 = rand_strided((6280, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_63 = rand_strided((8, 128, 28, 28), (100480, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cat_8 = rand_strided((8, 785, 128), (100480, 128, 1), device='cpu', dtype=torch.float32)
    getitem_35 = rand_strided((8, 785, 1), (785, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_8 = rand_strided((8, 785, 1), (785, 1, 1), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((6280, 128), (128, 1), device='cpu', dtype=torch.float32)
    slice_47 = rand_strided((8, 8, 784, 16), (301440, 16, 384, 1), device='cpu', dtype=torch.float32)
    getitem_39 = rand_strided((8, 32, 28, 28), (301440, 1, 10752, 384), device='cpu', dtype=torch.float32)
    getitem_40 = rand_strided((8, 48, 28, 28), (301440, 1, 10752, 384), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((8, 48, 28, 28), (301440, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cat_9 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    view_77 = rand_strided((6280, 128), (128, 1), device='cpu', dtype=torch.float32)
    mul_35 = rand_strided((8, 785, 128), (100480, 128, 1), device='cpu', dtype=torch.float32)
    view_79 = rand_strided((6280, 128), (128, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((6280, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_81 = rand_strided((6280, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    clone_31 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    mul_40 = rand_strided((8, 196, 320), (62720, 320, 1), device='cpu', dtype=torch.float32)
    view_85 = rand_strided((8, 320, 14, 14), (63040, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cat_11 = rand_strided((8, 197, 320), (63040, 320, 1), device='cpu', dtype=torch.float32)
    getitem_47 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_11 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    view_87 = rand_strided((1576, 320), (320, 1), device='cpu', dtype=torch.float32)
    slice_62 = rand_strided((8, 8, 196, 40), (189120, 40, 960, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((8, 80, 14, 14), (189120, 1, 13440, 960), device='cpu', dtype=torch.float32)
    getitem_52 = rand_strided((8, 120, 14, 14), (189120, 1, 13440, 960), device='cpu', dtype=torch.float32)
    getitem_53 = rand_strided((8, 120, 14, 14), (189120, 1, 13440, 960), device='cpu', dtype=torch.float32)
    cat_12 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    view_99 = rand_strided((1576, 320), (320, 1), device='cpu', dtype=torch.float32)
    mul_46 = rand_strided((8, 197, 320), (63040, 320, 1), device='cpu', dtype=torch.float32)
    view_101 = rand_strided((1576, 320), (320, 1), device='cpu', dtype=torch.float32)
    addmm_18 = rand_strided((1576, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    view_103 = rand_strided((1576, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    view_105 = rand_strided((8, 320, 14, 14), (63040, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cat_13 = rand_strided((8, 197, 320), (63040, 320, 1), device='cpu', dtype=torch.float32)
    getitem_57 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_13 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    view_107 = rand_strided((1576, 320), (320, 1), device='cpu', dtype=torch.float32)
    slice_74 = rand_strided((8, 8, 196, 40), (189120, 40, 960, 1), device='cpu', dtype=torch.float32)
    getitem_61 = rand_strided((8, 80, 14, 14), (189120, 1, 13440, 960), device='cpu', dtype=torch.float32)
    getitem_62 = rand_strided((8, 120, 14, 14), (189120, 1, 13440, 960), device='cpu', dtype=torch.float32)
    getitem_63 = rand_strided((8, 120, 14, 14), (189120, 1, 13440, 960), device='cpu', dtype=torch.float32)
    cat_14 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    view_119 = rand_strided((1576, 320), (320, 1), device='cpu', dtype=torch.float32)
    mul_55 = rand_strided((8, 197, 320), (63040, 320, 1), device='cpu', dtype=torch.float32)
    view_121 = rand_strided((1576, 320), (320, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((1576, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    view_123 = rand_strided((1576, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    clone_47 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    mul_60 = rand_strided((8, 49, 512), (25088, 512, 1), device='cpu', dtype=torch.float32)
    view_127 = rand_strided((8, 512, 7, 7), (25600, 1, 3584, 512), device='cpu', dtype=torch.float32)
    cat_16 = rand_strided((8, 50, 512), (25600, 512, 1), device='cpu', dtype=torch.float32)
    getitem_69 = rand_strided((8, 50, 1), (50, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_16 = rand_strided((8, 50, 1), (50, 1, 1), device='cpu', dtype=torch.float32)
    view_129 = rand_strided((400, 512), (512, 1), device='cpu', dtype=torch.float32)
    slice_89 = rand_strided((8, 8, 49, 64), (76800, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_73 = rand_strided((8, 128, 7, 7), (76800, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    getitem_74 = rand_strided((8, 192, 7, 7), (76800, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    getitem_75 = rand_strided((8, 192, 7, 7), (76800, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    cat_17 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    view_141 = rand_strided((400, 512), (512, 1), device='cpu', dtype=torch.float32)
    mul_66 = rand_strided((8, 50, 512), (25600, 512, 1), device='cpu', dtype=torch.float32)
    view_143 = rand_strided((400, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_26 = rand_strided((400, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_145 = rand_strided((400, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_147 = rand_strided((8, 512, 7, 7), (25600, 1, 3584, 512), device='cpu', dtype=torch.float32)
    cat_18 = rand_strided((8, 50, 512), (25600, 512, 1), device='cpu', dtype=torch.float32)
    getitem_79 = rand_strided((8, 50, 1), (50, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_18 = rand_strided((8, 50, 1), (50, 1, 1), device='cpu', dtype=torch.float32)
    view_149 = rand_strided((400, 512), (512, 1), device='cpu', dtype=torch.float32)
    slice_101 = rand_strided((8, 8, 49, 64), (76800, 64, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_83 = rand_strided((8, 128, 7, 7), (76800, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    getitem_84 = rand_strided((8, 192, 7, 7), (76800, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    getitem_85 = rand_strided((8, 192, 7, 7), (76800, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    cat_19 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    view_161 = rand_strided((400, 512), (512, 1), device='cpu', dtype=torch.float32)
    mul_75 = rand_strided((8, 50, 512), (25600, 512, 1), device='cpu', dtype=torch.float32)
    view_163 = rand_strided((400, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_30 = rand_strided((400, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_165 = rand_strided((400, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_80 = rand_strided((8, 50, 512), (25600, 512, 1), device='cpu', dtype=torch.float32)
    clone_64 = rand_strided((8, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_97 = rand_strided((1000, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_8 = rand_strided((8, 50, 1), (50, 1, 1), device='cpu', dtype=torch.float32)
    permute_101 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_105 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((8, 50, 1), (50, 1, 1), device='cpu', dtype=torch.float32)
    permute_109 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_116 = rand_strided((64, 64, 50), (3200, 1, 64), device='cpu', dtype=torch.float32)
    permute_117 = rand_strided((64, 64, 64), (4096, 1, 64), device='cpu', dtype=torch.float32)
    permute_118 = rand_strided((64, 50, 64), (3200, 64, 1), device='cpu', dtype=torch.float32)
    permute_119 = rand_strided((64, 64, 50), (3200, 1, 64), device='cpu', dtype=torch.float32)
    alias_8 = rand_strided((8, 8, 50, 64), (25600, 1, 512, 8), device='cpu', dtype=torch.float32)
    permute_122 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_128 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_132 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((8, 50, 1), (50, 1, 1), device='cpu', dtype=torch.float32)
    permute_136 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_143 = rand_strided((64, 64, 50), (3200, 1, 64), device='cpu', dtype=torch.float32)
    permute_144 = rand_strided((64, 64, 64), (4096, 1, 64), device='cpu', dtype=torch.float32)
    permute_145 = rand_strided((64, 50, 64), (3200, 64, 1), device='cpu', dtype=torch.float32)
    permute_146 = rand_strided((64, 64, 50), (3200, 1, 64), device='cpu', dtype=torch.float32)
    alias_9 = rand_strided((8, 8, 50, 64), (25600, 1, 512, 8), device='cpu', dtype=torch.float32)
    permute_149 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((8, 49, 1), (49, 1, 1), device='cpu', dtype=torch.float32)
    permute_157 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_161 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_165 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    permute_172 = rand_strided((64, 40, 197), (7880, 1, 40), device='cpu', dtype=torch.float32)
    permute_173 = rand_strided((64, 40, 40), (1600, 1, 40), device='cpu', dtype=torch.float32)
    permute_174 = rand_strided((64, 197, 40), (7880, 40, 1), device='cpu', dtype=torch.float32)
    permute_175 = rand_strided((64, 40, 197), (7880, 1, 40), device='cpu', dtype=torch.float32)
    alias_10 = rand_strided((8, 8, 197, 40), (63040, 1, 320, 8), device='cpu', dtype=torch.float32)
    permute_178 = rand_strided((960, 320), (320, 1), device='cpu', dtype=torch.float32)
    permute_184 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_188 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_192 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    permute_199 = rand_strided((64, 40, 197), (7880, 1, 40), device='cpu', dtype=torch.float32)
    permute_200 = rand_strided((64, 40, 40), (1600, 1, 40), device='cpu', dtype=torch.float32)
    permute_201 = rand_strided((64, 197, 40), (7880, 40, 1), device='cpu', dtype=torch.float32)
    permute_202 = rand_strided((64, 40, 197), (7880, 1, 40), device='cpu', dtype=torch.float32)
    alias_11 = rand_strided((8, 8, 197, 40), (63040, 1, 320, 8), device='cpu', dtype=torch.float32)
    permute_205 = rand_strided((960, 320), (320, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_213 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_217 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((8, 785, 1), (785, 1, 1), device='cpu', dtype=torch.float32)
    permute_221 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_228 = rand_strided((64, 16, 785), (12560, 1, 16), device='cpu', dtype=torch.float32)
    permute_229 = rand_strided((64, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    permute_230 = rand_strided((64, 785, 16), (12560, 16, 1), device='cpu', dtype=torch.float32)
    permute_231 = rand_strided((64, 16, 785), (12560, 1, 16), device='cpu', dtype=torch.float32)
    alias_12 = rand_strided((8, 8, 785, 16), (100480, 1, 128, 8), device='cpu', dtype=torch.float32)
    permute_234 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_240 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_244 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((8, 785, 1), (785, 1, 1), device='cpu', dtype=torch.float32)
    permute_248 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((64, 16, 785), (12560, 1, 16), device='cpu', dtype=torch.float32)
    permute_256 = rand_strided((64, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    permute_257 = rand_strided((64, 785, 16), (12560, 16, 1), device='cpu', dtype=torch.float32)
    permute_258 = rand_strided((64, 16, 785), (12560, 1, 16), device='cpu', dtype=torch.float32)
    alias_13 = rand_strided((8, 8, 785, 16), (100480, 1, 128, 8), device='cpu', dtype=torch.float32)
    permute_261 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((8, 784, 1), (784, 1, 1), device='cpu', dtype=torch.float32)
    permute_269 = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_273 = rand_strided((512, 64), (64, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((8, 3137, 1), (3137, 1, 1), device='cpu', dtype=torch.float32)
    permute_277 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    permute_284 = rand_strided((64, 8, 3137), (25096, 1, 8), device='cpu', dtype=torch.float32)
    permute_285 = rand_strided((64, 8, 8), (64, 1, 8), device='cpu', dtype=torch.float32)
    permute_286 = rand_strided((64, 3137, 8), (25096, 8, 1), device='cpu', dtype=torch.float32)
    permute_287 = rand_strided((64, 8, 3137), (25096, 1, 8), device='cpu', dtype=torch.float32)
    alias_14 = rand_strided((8, 8, 3137, 8), (200768, 1, 64, 8), device='cpu', dtype=torch.float32)
    permute_290 = rand_strided((192, 64), (64, 1), device='cpu', dtype=torch.float32)
    permute_296 = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_300 = rand_strided((512, 64), (64, 1), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((8, 3137, 1), (3137, 1, 1), device='cpu', dtype=torch.float32)
    permute_304 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    permute_311 = rand_strided((64, 8, 3137), (25096, 1, 8), device='cpu', dtype=torch.float32)
    permute_312 = rand_strided((64, 8, 8), (64, 1, 8), device='cpu', dtype=torch.float32)
    permute_313 = rand_strided((64, 3137, 8), (25096, 8, 1), device='cpu', dtype=torch.float32)
    permute_314 = rand_strided((64, 8, 3137), (25096, 1, 8), device='cpu', dtype=torch.float32)
    alias_15 = rand_strided((8, 8, 3137, 8), (200768, 1, 64, 8), device='cpu', dtype=torch.float32)
    permute_317 = rand_strided((192, 64), (64, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((8, 3136, 1), (3136, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_2, primals_4, primals_6, primals_8, primals_11, primals_13, primals_15, primals_17, primals_20, primals_22, primals_24, primals_26, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_67, primals_69, primals_71, primals_75, primals_77, primals_79, primals_95, primals_97, primals_99, primals_103, primals_105, primals_107, primals_123, primals_125, primals_127, primals_131, primals_133, primals_135, primals_153, mul, view_1, cat_1, getitem_3, rsqrt_1, view_3, slice_8, getitem_7, getitem_8, getitem_9, cat_2, view_15, mul_6, view_17, addmm_2, view_19, view_21, cat_3, getitem_13, rsqrt_3, view_23, slice_20, getitem_17, getitem_18, getitem_19, cat_4, view_35, mul_15, view_37, addmm_6, view_39, clone_15, mul_20, view_43, cat_6, getitem_25, rsqrt_6, view_45, slice_35, getitem_29, getitem_30, getitem_31, cat_7, view_57, mul_26, view_59, addmm_10, view_61, view_63, cat_8, getitem_35, rsqrt_8, view_65, slice_47, getitem_39, getitem_40, getitem_41, cat_9, view_77, mul_35, view_79, addmm_14, view_81, clone_31, mul_40, view_85, cat_11, getitem_47, rsqrt_11, view_87, slice_62, getitem_51, getitem_52, getitem_53, cat_12, view_99, mul_46, view_101, addmm_18, view_103, view_105, cat_13, getitem_57, rsqrt_13, view_107, slice_74, getitem_61, getitem_62, getitem_63, cat_14, view_119, mul_55, view_121, addmm_22, view_123, clone_47, mul_60, view_127, cat_16, getitem_69, rsqrt_16, view_129, slice_89, getitem_73, getitem_74, getitem_75, cat_17, view_141, mul_66, view_143, addmm_26, view_145, view_147, cat_18, getitem_79, rsqrt_18, view_149, slice_101, getitem_83, getitem_84, getitem_85, cat_19, view_161, mul_75, view_163, addmm_30, view_165, mul_80, clone_64, permute_97, div_8, permute_101, permute_105, div_9, permute_109, permute_116, permute_117, permute_118, permute_119, alias_8, permute_122, permute_128, permute_132, div_11, permute_136, permute_143, permute_144, permute_145, permute_146, alias_9, permute_149, div_13, permute_157, permute_161, div_14, permute_165, permute_172, permute_173, permute_174, permute_175, alias_10, permute_178, permute_184, permute_188, div_16, permute_192, permute_199, permute_200, permute_201, permute_202, alias_11, permute_205, div_18, permute_213, permute_217, div_19, permute_221, permute_228, permute_229, permute_230, permute_231, alias_12, permute_234, permute_240, permute_244, div_21, permute_248, permute_255, permute_256, permute_257, permute_258, alias_13, permute_261, div_23, permute_269, permute_273, div_24, permute_277, permute_284, permute_285, permute_286, permute_287, alias_14, permute_290, permute_296, permute_300, div_26, permute_304, permute_311, permute_312, permute_313, permute_314, alias_15, permute_317, div_28, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('coat_lite_mini', benchmark_compiled_module)
