
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


cpp_fused_convolution_0 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(36864L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (36864L*x1) + (110592L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (110592L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(73728L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2359296L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(73728L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9216L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x2) + (294912L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(9216.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9216L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (32L*x1) + (294912L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (32L*x1) + (294912L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(73728L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(73728L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7077888L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x2) + (221184L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2304.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(96L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (96L*x1) + (221184L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (96L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (96L*x1) + (221184L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2654208L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x2) + (331776L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2304.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (331776L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (331776L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2654208L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x2) + (82944L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (82944L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (82944L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1105920L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x2) + (138240L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (138240L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (138240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1105920L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x2) + (34560L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (34560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (34560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(552960L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x2) + (69120L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(552960L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x2) + (69120L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (80L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(552960L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x2) + (69120L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(552960L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x2) + (69120L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(774144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x2) + (96768L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(774144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x2) + (96768L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(774144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x2) + (96768L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(774144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x2) + (24192L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(36.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (24192L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (24192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(331776L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(36.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(331776L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(36.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(331776L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(36.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(331776L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(36.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(331776L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(36.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1280L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1280L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1280L*x2) + (46080L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1280L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(10240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(36.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (96, ), (1, ))
    assert_size_stride(arg7_1, (96, ), (1, ))
    assert_size_stride(arg8_1, (96, ), (1, ))
    assert_size_stride(arg9_1, (96, ), (1, ))
    assert_size_stride(arg10_1, (24, ), (1, ))
    assert_size_stride(arg11_1, (24, ), (1, ))
    assert_size_stride(arg12_1, (144, ), (1, ))
    assert_size_stride(arg13_1, (144, ), (1, ))
    assert_size_stride(arg14_1, (144, ), (1, ))
    assert_size_stride(arg15_1, (144, ), (1, ))
    assert_size_stride(arg16_1, (24, ), (1, ))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (144, ), (1, ))
    assert_size_stride(arg19_1, (144, ), (1, ))
    assert_size_stride(arg20_1, (144, ), (1, ))
    assert_size_stride(arg21_1, (144, ), (1, ))
    assert_size_stride(arg22_1, (40, ), (1, ))
    assert_size_stride(arg23_1, (40, ), (1, ))
    assert_size_stride(arg24_1, (240, ), (1, ))
    assert_size_stride(arg25_1, (240, ), (1, ))
    assert_size_stride(arg26_1, (240, ), (1, ))
    assert_size_stride(arg27_1, (240, ), (1, ))
    assert_size_stride(arg28_1, (40, ), (1, ))
    assert_size_stride(arg29_1, (40, ), (1, ))
    assert_size_stride(arg30_1, (240, ), (1, ))
    assert_size_stride(arg31_1, (240, ), (1, ))
    assert_size_stride(arg32_1, (240, ), (1, ))
    assert_size_stride(arg33_1, (240, ), (1, ))
    assert_size_stride(arg34_1, (80, ), (1, ))
    assert_size_stride(arg35_1, (80, ), (1, ))
    assert_size_stride(arg36_1, (480, ), (1, ))
    assert_size_stride(arg37_1, (480, ), (1, ))
    assert_size_stride(arg38_1, (480, ), (1, ))
    assert_size_stride(arg39_1, (480, ), (1, ))
    assert_size_stride(arg40_1, (80, ), (1, ))
    assert_size_stride(arg41_1, (80, ), (1, ))
    assert_size_stride(arg42_1, (480, ), (1, ))
    assert_size_stride(arg43_1, (480, ), (1, ))
    assert_size_stride(arg44_1, (480, ), (1, ))
    assert_size_stride(arg45_1, (480, ), (1, ))
    assert_size_stride(arg46_1, (80, ), (1, ))
    assert_size_stride(arg47_1, (80, ), (1, ))
    assert_size_stride(arg48_1, (480, ), (1, ))
    assert_size_stride(arg49_1, (480, ), (1, ))
    assert_size_stride(arg50_1, (480, ), (1, ))
    assert_size_stride(arg51_1, (480, ), (1, ))
    assert_size_stride(arg52_1, (80, ), (1, ))
    assert_size_stride(arg53_1, (80, ), (1, ))
    assert_size_stride(arg54_1, (480, ), (1, ))
    assert_size_stride(arg55_1, (480, ), (1, ))
    assert_size_stride(arg56_1, (480, ), (1, ))
    assert_size_stride(arg57_1, (480, ), (1, ))
    assert_size_stride(arg58_1, (112, ), (1, ))
    assert_size_stride(arg59_1, (112, ), (1, ))
    assert_size_stride(arg60_1, (672, ), (1, ))
    assert_size_stride(arg61_1, (672, ), (1, ))
    assert_size_stride(arg62_1, (672, ), (1, ))
    assert_size_stride(arg63_1, (672, ), (1, ))
    assert_size_stride(arg64_1, (112, ), (1, ))
    assert_size_stride(arg65_1, (112, ), (1, ))
    assert_size_stride(arg66_1, (672, ), (1, ))
    assert_size_stride(arg67_1, (672, ), (1, ))
    assert_size_stride(arg68_1, (672, ), (1, ))
    assert_size_stride(arg69_1, (672, ), (1, ))
    assert_size_stride(arg70_1, (112, ), (1, ))
    assert_size_stride(arg71_1, (112, ), (1, ))
    assert_size_stride(arg72_1, (672, ), (1, ))
    assert_size_stride(arg73_1, (672, ), (1, ))
    assert_size_stride(arg74_1, (672, ), (1, ))
    assert_size_stride(arg75_1, (672, ), (1, ))
    assert_size_stride(arg76_1, (112, ), (1, ))
    assert_size_stride(arg77_1, (112, ), (1, ))
    assert_size_stride(arg78_1, (672, ), (1, ))
    assert_size_stride(arg79_1, (672, ), (1, ))
    assert_size_stride(arg80_1, (672, ), (1, ))
    assert_size_stride(arg81_1, (672, ), (1, ))
    assert_size_stride(arg82_1, (192, ), (1, ))
    assert_size_stride(arg83_1, (192, ), (1, ))
    assert_size_stride(arg84_1, (1152, ), (1, ))
    assert_size_stride(arg85_1, (1152, ), (1, ))
    assert_size_stride(arg86_1, (1152, ), (1, ))
    assert_size_stride(arg87_1, (1152, ), (1, ))
    assert_size_stride(arg88_1, (192, ), (1, ))
    assert_size_stride(arg89_1, (192, ), (1, ))
    assert_size_stride(arg90_1, (1152, ), (1, ))
    assert_size_stride(arg91_1, (1152, ), (1, ))
    assert_size_stride(arg92_1, (1152, ), (1, ))
    assert_size_stride(arg93_1, (1152, ), (1, ))
    assert_size_stride(arg94_1, (192, ), (1, ))
    assert_size_stride(arg95_1, (192, ), (1, ))
    assert_size_stride(arg96_1, (1152, ), (1, ))
    assert_size_stride(arg97_1, (1152, ), (1, ))
    assert_size_stride(arg98_1, (1152, ), (1, ))
    assert_size_stride(arg99_1, (1152, ), (1, ))
    assert_size_stride(arg100_1, (192, ), (1, ))
    assert_size_stride(arg101_1, (192, ), (1, ))
    assert_size_stride(arg102_1, (1152, ), (1, ))
    assert_size_stride(arg103_1, (1152, ), (1, ))
    assert_size_stride(arg104_1, (1152, ), (1, ))
    assert_size_stride(arg105_1, (1152, ), (1, ))
    assert_size_stride(arg106_1, (192, ), (1, ))
    assert_size_stride(arg107_1, (192, ), (1, ))
    assert_size_stride(arg108_1, (1152, ), (1, ))
    assert_size_stride(arg109_1, (1152, ), (1, ))
    assert_size_stride(arg110_1, (1152, ), (1, ))
    assert_size_stride(arg111_1, (1152, ), (1, ))
    assert_size_stride(arg112_1, (320, ), (1, ))
    assert_size_stride(arg113_1, (320, ), (1, ))
    assert_size_stride(arg114_1, (1280, ), (1, ))
    assert_size_stride(arg115_1, (1280, ), (1, ))
    assert_size_stride(arg116_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg117_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg119_1, (8, ), (1, ))
    assert_size_stride(arg120_1, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg121_1, (32, ), (1, ))
    assert_size_stride(arg122_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg123_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg124_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg125_1, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg126_1, (4, ), (1, ))
    assert_size_stride(arg127_1, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(arg128_1, (96, ), (1, ))
    assert_size_stride(arg129_1, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg130_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg131_1, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg132_1, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg133_1, (6, ), (1, ))
    assert_size_stride(arg134_1, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg135_1, (144, ), (1, ))
    assert_size_stride(arg136_1, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg137_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg138_1, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg139_1, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg140_1, (6, ), (1, ))
    assert_size_stride(arg141_1, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg142_1, (144, ), (1, ))
    assert_size_stride(arg143_1, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg144_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg145_1, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg146_1, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg147_1, (10, ), (1, ))
    assert_size_stride(arg148_1, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(arg149_1, (240, ), (1, ))
    assert_size_stride(arg150_1, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg151_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg152_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg153_1, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg154_1, (10, ), (1, ))
    assert_size_stride(arg155_1, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(arg156_1, (240, ), (1, ))
    assert_size_stride(arg157_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg158_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg159_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg160_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg161_1, (20, ), (1, ))
    assert_size_stride(arg162_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg163_1, (480, ), (1, ))
    assert_size_stride(arg164_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg165_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg166_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg167_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg168_1, (20, ), (1, ))
    assert_size_stride(arg169_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg170_1, (480, ), (1, ))
    assert_size_stride(arg171_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg172_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg173_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg174_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg175_1, (20, ), (1, ))
    assert_size_stride(arg176_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg177_1, (480, ), (1, ))
    assert_size_stride(arg178_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg179_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg180_1, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg181_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg182_1, (20, ), (1, ))
    assert_size_stride(arg183_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg184_1, (480, ), (1, ))
    assert_size_stride(arg185_1, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg186_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg187_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg188_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg189_1, (28, ), (1, ))
    assert_size_stride(arg190_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg191_1, (672, ), (1, ))
    assert_size_stride(arg192_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg193_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg194_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg195_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg196_1, (28, ), (1, ))
    assert_size_stride(arg197_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg198_1, (672, ), (1, ))
    assert_size_stride(arg199_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg200_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg201_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg202_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg203_1, (28, ), (1, ))
    assert_size_stride(arg204_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg205_1, (672, ), (1, ))
    assert_size_stride(arg206_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg207_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg208_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg209_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg210_1, (28, ), (1, ))
    assert_size_stride(arg211_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg212_1, (672, ), (1, ))
    assert_size_stride(arg213_1, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg214_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg215_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg216_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg217_1, (48, ), (1, ))
    assert_size_stride(arg218_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg219_1, (1152, ), (1, ))
    assert_size_stride(arg220_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg221_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg222_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg223_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg224_1, (48, ), (1, ))
    assert_size_stride(arg225_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg226_1, (1152, ), (1, ))
    assert_size_stride(arg227_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg228_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg229_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg230_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg231_1, (48, ), (1, ))
    assert_size_stride(arg232_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg233_1, (1152, ), (1, ))
    assert_size_stride(arg234_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg235_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg236_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg237_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg238_1, (48, ), (1, ))
    assert_size_stride(arg239_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg240_1, (1152, ), (1, ))
    assert_size_stride(arg241_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg242_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg243_1, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg244_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg245_1, (48, ), (1, ))
    assert_size_stride(arg246_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg247_1, (1152, ), (1, ))
    assert_size_stride(arg248_1, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg249_1, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg250_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg251_1, (1000, ), (1, ))
    assert_size_stride(arg252_1, (32, ), (1, ))
    assert_size_stride(arg253_1, (32, ), (1, ))
    assert_size_stride(arg254_1, (32, ), (1, ))
    assert_size_stride(arg255_1, (32, ), (1, ))
    assert_size_stride(arg256_1, (16, ), (1, ))
    assert_size_stride(arg257_1, (16, ), (1, ))
    assert_size_stride(arg258_1, (96, ), (1, ))
    assert_size_stride(arg259_1, (96, ), (1, ))
    assert_size_stride(arg260_1, (96, ), (1, ))
    assert_size_stride(arg261_1, (96, ), (1, ))
    assert_size_stride(arg262_1, (24, ), (1, ))
    assert_size_stride(arg263_1, (24, ), (1, ))
    assert_size_stride(arg264_1, (144, ), (1, ))
    assert_size_stride(arg265_1, (144, ), (1, ))
    assert_size_stride(arg266_1, (144, ), (1, ))
    assert_size_stride(arg267_1, (144, ), (1, ))
    assert_size_stride(arg268_1, (24, ), (1, ))
    assert_size_stride(arg269_1, (24, ), (1, ))
    assert_size_stride(arg270_1, (144, ), (1, ))
    assert_size_stride(arg271_1, (144, ), (1, ))
    assert_size_stride(arg272_1, (144, ), (1, ))
    assert_size_stride(arg273_1, (144, ), (1, ))
    assert_size_stride(arg274_1, (40, ), (1, ))
    assert_size_stride(arg275_1, (40, ), (1, ))
    assert_size_stride(arg276_1, (240, ), (1, ))
    assert_size_stride(arg277_1, (240, ), (1, ))
    assert_size_stride(arg278_1, (240, ), (1, ))
    assert_size_stride(arg279_1, (240, ), (1, ))
    assert_size_stride(arg280_1, (40, ), (1, ))
    assert_size_stride(arg281_1, (40, ), (1, ))
    assert_size_stride(arg282_1, (240, ), (1, ))
    assert_size_stride(arg283_1, (240, ), (1, ))
    assert_size_stride(arg284_1, (240, ), (1, ))
    assert_size_stride(arg285_1, (240, ), (1, ))
    assert_size_stride(arg286_1, (80, ), (1, ))
    assert_size_stride(arg287_1, (80, ), (1, ))
    assert_size_stride(arg288_1, (480, ), (1, ))
    assert_size_stride(arg289_1, (480, ), (1, ))
    assert_size_stride(arg290_1, (480, ), (1, ))
    assert_size_stride(arg291_1, (480, ), (1, ))
    assert_size_stride(arg292_1, (80, ), (1, ))
    assert_size_stride(arg293_1, (80, ), (1, ))
    assert_size_stride(arg294_1, (480, ), (1, ))
    assert_size_stride(arg295_1, (480, ), (1, ))
    assert_size_stride(arg296_1, (480, ), (1, ))
    assert_size_stride(arg297_1, (480, ), (1, ))
    assert_size_stride(arg298_1, (80, ), (1, ))
    assert_size_stride(arg299_1, (80, ), (1, ))
    assert_size_stride(arg300_1, (480, ), (1, ))
    assert_size_stride(arg301_1, (480, ), (1, ))
    assert_size_stride(arg302_1, (480, ), (1, ))
    assert_size_stride(arg303_1, (480, ), (1, ))
    assert_size_stride(arg304_1, (80, ), (1, ))
    assert_size_stride(arg305_1, (80, ), (1, ))
    assert_size_stride(arg306_1, (480, ), (1, ))
    assert_size_stride(arg307_1, (480, ), (1, ))
    assert_size_stride(arg308_1, (480, ), (1, ))
    assert_size_stride(arg309_1, (480, ), (1, ))
    assert_size_stride(arg310_1, (112, ), (1, ))
    assert_size_stride(arg311_1, (112, ), (1, ))
    assert_size_stride(arg312_1, (672, ), (1, ))
    assert_size_stride(arg313_1, (672, ), (1, ))
    assert_size_stride(arg314_1, (672, ), (1, ))
    assert_size_stride(arg315_1, (672, ), (1, ))
    assert_size_stride(arg316_1, (112, ), (1, ))
    assert_size_stride(arg317_1, (112, ), (1, ))
    assert_size_stride(arg318_1, (672, ), (1, ))
    assert_size_stride(arg319_1, (672, ), (1, ))
    assert_size_stride(arg320_1, (672, ), (1, ))
    assert_size_stride(arg321_1, (672, ), (1, ))
    assert_size_stride(arg322_1, (112, ), (1, ))
    assert_size_stride(arg323_1, (112, ), (1, ))
    assert_size_stride(arg324_1, (672, ), (1, ))
    assert_size_stride(arg325_1, (672, ), (1, ))
    assert_size_stride(arg326_1, (672, ), (1, ))
    assert_size_stride(arg327_1, (672, ), (1, ))
    assert_size_stride(arg328_1, (112, ), (1, ))
    assert_size_stride(arg329_1, (112, ), (1, ))
    assert_size_stride(arg330_1, (672, ), (1, ))
    assert_size_stride(arg331_1, (672, ), (1, ))
    assert_size_stride(arg332_1, (672, ), (1, ))
    assert_size_stride(arg333_1, (672, ), (1, ))
    assert_size_stride(arg334_1, (192, ), (1, ))
    assert_size_stride(arg335_1, (192, ), (1, ))
    assert_size_stride(arg336_1, (1152, ), (1, ))
    assert_size_stride(arg337_1, (1152, ), (1, ))
    assert_size_stride(arg338_1, (1152, ), (1, ))
    assert_size_stride(arg339_1, (1152, ), (1, ))
    assert_size_stride(arg340_1, (192, ), (1, ))
    assert_size_stride(arg341_1, (192, ), (1, ))
    assert_size_stride(arg342_1, (1152, ), (1, ))
    assert_size_stride(arg343_1, (1152, ), (1, ))
    assert_size_stride(arg344_1, (1152, ), (1, ))
    assert_size_stride(arg345_1, (1152, ), (1, ))
    assert_size_stride(arg346_1, (192, ), (1, ))
    assert_size_stride(arg347_1, (192, ), (1, ))
    assert_size_stride(arg348_1, (1152, ), (1, ))
    assert_size_stride(arg349_1, (1152, ), (1, ))
    assert_size_stride(arg350_1, (1152, ), (1, ))
    assert_size_stride(arg351_1, (1152, ), (1, ))
    assert_size_stride(arg352_1, (192, ), (1, ))
    assert_size_stride(arg353_1, (192, ), (1, ))
    assert_size_stride(arg354_1, (1152, ), (1, ))
    assert_size_stride(arg355_1, (1152, ), (1, ))
    assert_size_stride(arg356_1, (1152, ), (1, ))
    assert_size_stride(arg357_1, (1152, ), (1, ))
    assert_size_stride(arg358_1, (192, ), (1, ))
    assert_size_stride(arg359_1, (192, ), (1, ))
    assert_size_stride(arg360_1, (1152, ), (1, ))
    assert_size_stride(arg361_1, (1152, ), (1, ))
    assert_size_stride(arg362_1, (1152, ), (1, ))
    assert_size_stride(arg363_1, (1152, ), (1, ))
    assert_size_stride(arg364_1, (320, ), (1, ))
    assert_size_stride(arg365_1, (320, ), (1, ))
    assert_size_stride(arg366_1, (1280, ), (1, ))
    assert_size_stride(arg367_1, (1280, ), (1, ))
    assert_size_stride(arg368_1, (8, 3, 192, 192), (110592, 36864, 192, 1))
    buf0 = empty_strided((8, 3, 192, 192), (110592, 1, 576, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg368_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg116_1
    del arg368_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 32, 96, 96), (294912, 1, 3072, 32))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = buf3; del buf3  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_1(c_void_p(buf4.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg0_1
    del arg1_1
    del arg252_1
    del arg253_1
    # Source Nodes: [shortcut, x_5], Original ATen: [aten.convolution, aten.silu]
    buf5 = extern_kernels.convolution(buf4, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf5, (8, 32, 96, 96), (294912, 1, 3072, 32))
    del arg117_1
    del buf4
    buf6 = buf5; del buf5  # reuse
    buf7 = empty_strided((8, 32, 1, 1), (32, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf8 = reinterpret_tensor(buf7, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf7  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_2(c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()))
    del arg254_1
    del arg255_1
    del arg2_1
    del arg3_1
    # Source Nodes: [x_9, x_se, x_se_1], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf9 = extern_kernels.convolution(buf8, arg118_1, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf9, (8, 8, 1, 1), (8, 1, 8, 8))
    del arg118_1
    del arg119_1
    del buf8
    buf10 = buf9; del buf9  # reuse
    cpp_fused_silu_3(c_void_p(buf10.data_ptr()))
    # Source Nodes: [x_se_2, x_se_3], Original ATen: [aten.convolution, aten.silu]
    buf11 = extern_kernels.convolution(buf10, arg120_1, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf11, (8, 32, 1, 1), (32, 1, 32, 32))
    del arg120_1
    del arg121_1
    del buf10
    buf12 = buf6; del buf6  # reuse
    cpp_fused_mul_sigmoid_silu_4(c_void_p(buf12.data_ptr()), c_void_p(buf11.data_ptr()))
    del buf11
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_11, x_9], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf13 = extern_kernels.convolution(buf12, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf13, (8, 16, 96, 96), (147456, 1, 1536, 16))
    del arg122_1
    del buf12
    buf14 = buf13; del buf13  # reuse
    cpp_fused__native_batch_norm_legit_no_training_5(c_void_p(buf14.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg256_1
    del arg257_1
    del arg4_1
    del arg5_1
    # Source Nodes: [x_12, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf15 = extern_kernels.convolution(buf14, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf15, (8, 96, 96, 96), (884736, 1, 9216, 96))
    del arg123_1
    del buf14
    buf16 = buf15; del buf15  # reuse
    buf17 = buf16; del buf16  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_6(c_void_p(buf17.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()))
    del arg258_1
    del arg259_1
    del arg6_1
    del arg7_1
    # Source Nodes: [x_20, x_21], Original ATen: [aten.convolution, aten.silu]
    buf18 = extern_kernels.convolution(buf17, arg124_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
    assert_size_stride(buf18, (8, 96, 48, 48), (221184, 1, 4608, 96))
    del arg124_1
    del buf17
    buf19 = buf18; del buf18  # reuse
    buf20 = empty_strided((8, 96, 1, 1), (96, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf21 = reinterpret_tensor(buf20, (8, 96, 1, 1), (96, 1, 96, 96), 0); del buf20  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_7(c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg260_1
    del arg261_1
    del arg8_1
    del arg9_1
    # Source Nodes: [x_25, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf22 = extern_kernels.convolution(buf21, arg125_1, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf22, (8, 4, 1, 1), (4, 1, 4, 4))
    del arg125_1
    del arg126_1
    del buf21
    buf23 = buf22; del buf22  # reuse
    cpp_fused_silu_8(c_void_p(buf23.data_ptr()))
    # Source Nodes: [x_se_6, x_se_7], Original ATen: [aten.convolution, aten.silu]
    buf24 = extern_kernels.convolution(buf23, arg127_1, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf24, (8, 96, 1, 1), (96, 1, 96, 96))
    del arg127_1
    del arg128_1
    del buf23
    buf25 = buf19; del buf19  # reuse
    cpp_fused_mul_sigmoid_silu_9(c_void_p(buf25.data_ptr()), c_void_p(buf24.data_ptr()))
    del buf24
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25, x_26, x_27], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf26 = extern_kernels.convolution(buf25, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf26, (8, 24, 48, 48), (55296, 1, 1152, 24))
    del arg129_1
    del buf25
    buf27 = buf26; del buf26  # reuse
    cpp_fused__native_batch_norm_legit_no_training_10(c_void_p(buf27.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()))
    del arg10_1
    del arg11_1
    del arg262_1
    del arg263_1
    # Source Nodes: [x_32], Original ATen: [aten.convolution]
    buf28 = extern_kernels.convolution(buf27, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf28, (8, 144, 48, 48), (331776, 1, 6912, 144))
    del arg130_1
    buf29 = buf28; del buf28  # reuse
    buf30 = buf29; del buf29  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_11(c_void_p(buf30.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()))
    del arg12_1
    del arg13_1
    del arg264_1
    del arg265_1
    # Source Nodes: [x_36, x_37], Original ATen: [aten.convolution, aten.silu]
    buf31 = extern_kernels.convolution(buf30, arg131_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
    assert_size_stride(buf31, (8, 144, 48, 48), (331776, 1, 6912, 144))
    del arg131_1
    del buf30
    buf32 = buf31; del buf31  # reuse
    buf33 = empty_strided((8, 144, 1, 1), (144, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    buf34 = reinterpret_tensor(buf33, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf33  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_12(c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()))
    del arg14_1
    del arg15_1
    del arg266_1
    del arg267_1
    # Source Nodes: [x_41, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf35 = extern_kernels.convolution(buf34, arg132_1, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf35, (8, 6, 1, 1), (6, 1, 6, 6))
    del arg132_1
    del arg133_1
    del buf34
    buf36 = buf35; del buf35  # reuse
    cpp_fused_silu_13(c_void_p(buf36.data_ptr()))
    # Source Nodes: [x_se_10, x_se_11], Original ATen: [aten.convolution, aten.silu]
    buf37 = extern_kernels.convolution(buf36, arg134_1, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf37, (8, 144, 1, 1), (144, 1, 144, 144))
    del arg134_1
    del arg135_1
    del buf36
    buf38 = buf32; del buf32  # reuse
    cpp_fused_mul_sigmoid_silu_14(c_void_p(buf38.data_ptr()), c_void_p(buf37.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41, x_42, x_43], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf39 = extern_kernels.convolution(buf38, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf39, (8, 24, 48, 48), (55296, 1, 1152, 24))
    del arg136_1
    del buf38
    buf40 = buf27; del buf27  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_15(c_void_p(buf40.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg16_1
    del arg17_1
    del arg268_1
    del arg269_1
    del buf39
    # Source Nodes: [shortcut_3, x_44, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf41 = extern_kernels.convolution(buf40, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf41, (8, 144, 48, 48), (331776, 1, 6912, 144))
    del arg137_1
    del buf40
    buf42 = buf41; del buf41  # reuse
    buf43 = buf42; del buf42  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_16(c_void_p(buf43.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()))
    del arg18_1
    del arg19_1
    del arg270_1
    del arg271_1
    # Source Nodes: [x_53, x_54], Original ATen: [aten.convolution, aten.silu]
    buf44 = extern_kernels.convolution(buf43, arg138_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
    assert_size_stride(buf44, (8, 144, 24, 24), (82944, 1, 3456, 144))
    del arg138_1
    del buf43
    buf45 = buf44; del buf44  # reuse
    buf46 = reinterpret_tensor(buf37, (8, 144, 1, 1), (144, 1, 1152, 1152), 0); del buf37  # reuse
    buf47 = reinterpret_tensor(buf46, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf46  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_17(c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()))
    del arg20_1
    del arg21_1
    del arg272_1
    del arg273_1
    # Source Nodes: [x_58, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf48 = extern_kernels.convolution(buf47, arg139_1, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf48, (8, 6, 1, 1), (6, 1, 6, 6))
    del arg139_1
    del arg140_1
    del buf47
    buf49 = buf48; del buf48  # reuse
    cpp_fused_silu_18(c_void_p(buf49.data_ptr()))
    # Source Nodes: [x_se_14, x_se_15], Original ATen: [aten.convolution, aten.silu]
    buf50 = extern_kernels.convolution(buf49, arg141_1, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf50, (8, 144, 1, 1), (144, 1, 144, 144))
    del arg141_1
    del arg142_1
    del buf49
    buf51 = buf45; del buf45  # reuse
    cpp_fused_mul_sigmoid_silu_19(c_void_p(buf51.data_ptr()), c_void_p(buf50.data_ptr()))
    del buf50
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58, x_59, x_60], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf52 = extern_kernels.convolution(buf51, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf52, (8, 40, 24, 24), (23040, 1, 960, 40))
    del arg143_1
    del buf51
    buf53 = buf52; del buf52  # reuse
    cpp_fused__native_batch_norm_legit_no_training_20(c_void_p(buf53.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg22_1
    del arg23_1
    del arg274_1
    del arg275_1
    # Source Nodes: [x_65], Original ATen: [aten.convolution]
    buf54 = extern_kernels.convolution(buf53, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf54, (8, 240, 24, 24), (138240, 1, 5760, 240))
    del arg144_1
    buf55 = buf54; del buf54  # reuse
    buf56 = buf55; del buf55  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_21(c_void_p(buf56.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()))
    del arg24_1
    del arg25_1
    del arg276_1
    del arg277_1
    # Source Nodes: [x_69, x_70], Original ATen: [aten.convolution, aten.silu]
    buf57 = extern_kernels.convolution(buf56, arg145_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf57, (8, 240, 24, 24), (138240, 1, 5760, 240))
    del arg145_1
    del buf56
    buf58 = buf57; del buf57  # reuse
    buf59 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf60 = reinterpret_tensor(buf59, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf59  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_22(c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()))
    del arg26_1
    del arg278_1
    del arg279_1
    del arg27_1
    # Source Nodes: [x_74, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf61 = extern_kernels.convolution(buf60, arg146_1, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf61, (8, 10, 1, 1), (10, 1, 10, 10))
    del arg146_1
    del arg147_1
    del buf60
    buf62 = buf61; del buf61  # reuse
    cpp_fused_silu_23(c_void_p(buf62.data_ptr()))
    # Source Nodes: [x_se_18, x_se_19], Original ATen: [aten.convolution, aten.silu]
    buf63 = extern_kernels.convolution(buf62, arg148_1, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf63, (8, 240, 1, 1), (240, 1, 240, 240))
    del arg148_1
    del arg149_1
    del buf62
    buf64 = buf58; del buf58  # reuse
    cpp_fused_mul_sigmoid_silu_24(c_void_p(buf64.data_ptr()), c_void_p(buf63.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74, x_75, x_76], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf65 = extern_kernels.convolution(buf64, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf65, (8, 40, 24, 24), (23040, 1, 960, 40))
    del arg150_1
    del buf64
    buf66 = buf53; del buf53  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_25(c_void_p(buf66.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg280_1
    del arg281_1
    del arg28_1
    del arg29_1
    del buf65
    # Source Nodes: [shortcut_5, x_77, x_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf67 = extern_kernels.convolution(buf66, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf67, (8, 240, 24, 24), (138240, 1, 5760, 240))
    del arg151_1
    del buf66
    buf68 = buf67; del buf67  # reuse
    buf69 = buf68; del buf68  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_26(c_void_p(buf69.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()))
    del arg282_1
    del arg283_1
    del arg30_1
    del arg31_1
    # Source Nodes: [x_86, x_87], Original ATen: [aten.convolution, aten.silu]
    buf70 = extern_kernels.convolution(buf69, arg152_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf70, (8, 240, 12, 12), (34560, 1, 2880, 240))
    del arg152_1
    del buf69
    buf71 = buf70; del buf70  # reuse
    buf72 = reinterpret_tensor(buf63, (8, 240, 1, 1), (240, 1, 1920, 1920), 0); del buf63  # reuse
    buf73 = reinterpret_tensor(buf72, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf72  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_27(c_void_p(buf71.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()))
    del arg284_1
    del arg285_1
    del arg32_1
    del arg33_1
    # Source Nodes: [x_91, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf74 = extern_kernels.convolution(buf73, arg153_1, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf74, (8, 10, 1, 1), (10, 1, 10, 10))
    del arg153_1
    del arg154_1
    del buf73
    buf75 = buf74; del buf74  # reuse
    cpp_fused_silu_28(c_void_p(buf75.data_ptr()))
    # Source Nodes: [x_se_22, x_se_23], Original ATen: [aten.convolution, aten.silu]
    buf76 = extern_kernels.convolution(buf75, arg155_1, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf76, (8, 240, 1, 1), (240, 1, 240, 240))
    del arg155_1
    del arg156_1
    del buf75
    buf77 = buf71; del buf71  # reuse
    cpp_fused_mul_sigmoid_silu_29(c_void_p(buf77.data_ptr()), c_void_p(buf76.data_ptr()))
    del buf76
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91, x_92, x_93], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf78 = extern_kernels.convolution(buf77, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf78, (8, 80, 12, 12), (11520, 1, 960, 80))
    del arg157_1
    del buf77
    buf79 = buf78; del buf78  # reuse
    cpp_fused__native_batch_norm_legit_no_training_30(c_void_p(buf79.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg286_1
    del arg287_1
    del arg34_1
    del arg35_1
    # Source Nodes: [x_98], Original ATen: [aten.convolution]
    buf80 = extern_kernels.convolution(buf79, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf80, (8, 480, 12, 12), (69120, 1, 5760, 480))
    del arg158_1
    buf81 = buf80; del buf80  # reuse
    buf82 = buf81; del buf81  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_31(c_void_p(buf82.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()))
    del arg288_1
    del arg289_1
    del arg36_1
    del arg37_1
    # Source Nodes: [x_102, x_103], Original ATen: [aten.convolution, aten.silu]
    buf83 = extern_kernels.convolution(buf82, arg159_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf83, (8, 480, 12, 12), (69120, 1, 5760, 480))
    del arg159_1
    del buf82
    buf84 = buf83; del buf83  # reuse
    buf85 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cpu', dtype=torch.float32)
    buf86 = reinterpret_tensor(buf85, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf85  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_32(c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()))
    del arg290_1
    del arg291_1
    del arg38_1
    del arg39_1
    # Source Nodes: [x_107, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf87 = extern_kernels.convolution(buf86, arg160_1, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf87, (8, 20, 1, 1), (20, 1, 20, 20))
    del arg160_1
    del arg161_1
    del buf86
    buf88 = buf87; del buf87  # reuse
    cpp_fused_silu_33(c_void_p(buf88.data_ptr()))
    # Source Nodes: [x_se_26, x_se_27], Original ATen: [aten.convolution, aten.silu]
    buf89 = extern_kernels.convolution(buf88, arg162_1, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf89, (8, 480, 1, 1), (480, 1, 480, 480))
    del arg162_1
    del arg163_1
    del buf88
    buf90 = buf84; del buf84  # reuse
    cpp_fused_mul_sigmoid_silu_34(c_void_p(buf90.data_ptr()), c_void_p(buf89.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_107, x_108, x_109], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf91 = extern_kernels.convolution(buf90, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf91, (8, 80, 12, 12), (11520, 1, 960, 80))
    del arg164_1
    del buf90
    buf92 = buf79; del buf79  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_35(c_void_p(buf92.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg292_1
    del arg293_1
    del arg40_1
    del arg41_1
    del buf91
    # Source Nodes: [x_115], Original ATen: [aten.convolution]
    buf93 = extern_kernels.convolution(buf92, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf93, (8, 480, 12, 12), (69120, 1, 5760, 480))
    del arg165_1
    buf94 = buf93; del buf93  # reuse
    buf95 = buf94; del buf94  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_36(c_void_p(buf95.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()))
    del arg294_1
    del arg295_1
    del arg42_1
    del arg43_1
    # Source Nodes: [x_119, x_120], Original ATen: [aten.convolution, aten.silu]
    buf96 = extern_kernels.convolution(buf95, arg166_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf96, (8, 480, 12, 12), (69120, 1, 5760, 480))
    del arg166_1
    del buf95
    buf97 = buf96; del buf96  # reuse
    buf98 = reinterpret_tensor(buf89, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf89  # reuse
    buf99 = reinterpret_tensor(buf98, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf98  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_37(c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()))
    del arg296_1
    del arg297_1
    del arg44_1
    del arg45_1
    # Source Nodes: [x_124, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf100 = extern_kernels.convolution(buf99, arg167_1, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf100, (8, 20, 1, 1), (20, 1, 20, 20))
    del arg167_1
    del arg168_1
    del buf99
    buf101 = buf100; del buf100  # reuse
    cpp_fused_silu_38(c_void_p(buf101.data_ptr()))
    # Source Nodes: [x_se_30, x_se_31], Original ATen: [aten.convolution, aten.silu]
    buf102 = extern_kernels.convolution(buf101, arg169_1, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf102, (8, 480, 1, 1), (480, 1, 480, 480))
    del arg169_1
    del arg170_1
    del buf101
    buf103 = buf97; del buf97  # reuse
    cpp_fused_mul_sigmoid_silu_39(c_void_p(buf103.data_ptr()), c_void_p(buf102.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_124, x_125, x_126], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf104 = extern_kernels.convolution(buf103, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf104, (8, 80, 12, 12), (11520, 1, 960, 80))
    del arg171_1
    del buf103
    buf105 = buf104; del buf104  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_40(c_void_p(buf105.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf92.data_ptr()))
    del arg298_1
    del arg299_1
    del arg46_1
    del arg47_1
    del buf92
    # Source Nodes: [x_132], Original ATen: [aten.convolution]
    buf106 = extern_kernels.convolution(buf105, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf106, (8, 480, 12, 12), (69120, 1, 5760, 480))
    del arg172_1
    buf107 = buf106; del buf106  # reuse
    buf108 = buf107; del buf107  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_41(c_void_p(buf108.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()))
    del arg300_1
    del arg301_1
    del arg48_1
    del arg49_1
    # Source Nodes: [x_136, x_137], Original ATen: [aten.convolution, aten.silu]
    buf109 = extern_kernels.convolution(buf108, arg173_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf109, (8, 480, 12, 12), (69120, 1, 5760, 480))
    del arg173_1
    del buf108
    buf110 = buf109; del buf109  # reuse
    buf111 = reinterpret_tensor(buf102, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf102  # reuse
    buf112 = reinterpret_tensor(buf111, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf111  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_42(c_void_p(buf110.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()))
    del arg302_1
    del arg303_1
    del arg50_1
    del arg51_1
    # Source Nodes: [x_141, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf113 = extern_kernels.convolution(buf112, arg174_1, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf113, (8, 20, 1, 1), (20, 1, 20, 20))
    del arg174_1
    del arg175_1
    del buf112
    buf114 = buf113; del buf113  # reuse
    cpp_fused_silu_43(c_void_p(buf114.data_ptr()))
    # Source Nodes: [x_se_34, x_se_35], Original ATen: [aten.convolution, aten.silu]
    buf115 = extern_kernels.convolution(buf114, arg176_1, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf115, (8, 480, 1, 1), (480, 1, 480, 480))
    del arg176_1
    del arg177_1
    del buf114
    buf116 = buf110; del buf110  # reuse
    cpp_fused_mul_sigmoid_silu_44(c_void_p(buf116.data_ptr()), c_void_p(buf115.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate, x_141, x_142, x_143], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf117 = extern_kernels.convolution(buf116, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf117, (8, 80, 12, 12), (11520, 1, 960, 80))
    del arg178_1
    del buf116
    buf118 = buf105; del buf105  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_45(c_void_p(buf118.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg304_1
    del arg305_1
    del arg52_1
    del arg53_1
    del buf117
    # Source Nodes: [shortcut_9, x_144, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf119 = extern_kernels.convolution(buf118, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf119, (8, 480, 12, 12), (69120, 1, 5760, 480))
    del arg179_1
    del buf118
    buf120 = buf119; del buf119  # reuse
    buf121 = buf120; del buf120  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_46(c_void_p(buf121.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()))
    del arg306_1
    del arg307_1
    del arg54_1
    del arg55_1
    # Source Nodes: [x_153, x_154], Original ATen: [aten.convolution, aten.silu]
    buf122 = extern_kernels.convolution(buf121, arg180_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf122, (8, 480, 12, 12), (69120, 1, 5760, 480))
    del arg180_1
    del buf121
    buf123 = buf122; del buf122  # reuse
    buf124 = reinterpret_tensor(buf115, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf115  # reuse
    buf125 = reinterpret_tensor(buf124, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf124  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_47(c_void_p(buf123.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()))
    del arg308_1
    del arg309_1
    del arg56_1
    del arg57_1
    # Source Nodes: [x_158, x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf126 = extern_kernels.convolution(buf125, arg181_1, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf126, (8, 20, 1, 1), (20, 1, 20, 20))
    del arg181_1
    del arg182_1
    del buf125
    buf127 = buf126; del buf126  # reuse
    cpp_fused_silu_48(c_void_p(buf127.data_ptr()))
    # Source Nodes: [x_se_38, x_se_39], Original ATen: [aten.convolution, aten.silu]
    buf128 = extern_kernels.convolution(buf127, arg183_1, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf128, (8, 480, 1, 1), (480, 1, 480, 480))
    del arg183_1
    del arg184_1
    del buf127
    buf129 = buf123; del buf123  # reuse
    cpp_fused_mul_sigmoid_silu_49(c_void_p(buf129.data_ptr()), c_void_p(buf128.data_ptr()))
    del buf128
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_158, x_159, x_160], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf130 = extern_kernels.convolution(buf129, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf130, (8, 112, 12, 12), (16128, 1, 1344, 112))
    del arg185_1
    del buf129
    buf131 = buf130; del buf130  # reuse
    cpp_fused__native_batch_norm_legit_no_training_50(c_void_p(buf131.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg310_1
    del arg311_1
    del arg58_1
    del arg59_1
    # Source Nodes: [x_165], Original ATen: [aten.convolution]
    buf132 = extern_kernels.convolution(buf131, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf132, (8, 672, 12, 12), (96768, 1, 8064, 672))
    del arg186_1
    buf133 = buf132; del buf132  # reuse
    buf134 = buf133; del buf133  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_51(c_void_p(buf134.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()))
    del arg312_1
    del arg313_1
    del arg60_1
    del arg61_1
    # Source Nodes: [x_169, x_170], Original ATen: [aten.convolution, aten.silu]
    buf135 = extern_kernels.convolution(buf134, arg187_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf135, (8, 672, 12, 12), (96768, 1, 8064, 672))
    del arg187_1
    del buf134
    buf136 = buf135; del buf135  # reuse
    buf137 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cpu', dtype=torch.float32)
    buf138 = reinterpret_tensor(buf137, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf137  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_52(c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()))
    del arg314_1
    del arg315_1
    del arg62_1
    del arg63_1
    # Source Nodes: [x_174, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf139 = extern_kernels.convolution(buf138, arg188_1, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf139, (8, 28, 1, 1), (28, 1, 28, 28))
    del arg188_1
    del arg189_1
    del buf138
    buf140 = buf139; del buf139  # reuse
    cpp_fused_silu_53(c_void_p(buf140.data_ptr()))
    # Source Nodes: [x_se_42, x_se_43], Original ATen: [aten.convolution, aten.silu]
    buf141 = extern_kernels.convolution(buf140, arg190_1, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf141, (8, 672, 1, 1), (672, 1, 672, 672))
    del arg190_1
    del arg191_1
    del buf140
    buf142 = buf136; del buf136  # reuse
    cpp_fused_mul_sigmoid_silu_54(c_void_p(buf142.data_ptr()), c_void_p(buf141.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_174, x_175, x_176], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf143 = extern_kernels.convolution(buf142, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf143, (8, 112, 12, 12), (16128, 1, 1344, 112))
    del arg192_1
    del buf142
    buf144 = buf131; del buf131  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_55(c_void_p(buf144.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg316_1
    del arg317_1
    del arg64_1
    del arg65_1
    del buf143
    # Source Nodes: [x_182], Original ATen: [aten.convolution]
    buf145 = extern_kernels.convolution(buf144, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf145, (8, 672, 12, 12), (96768, 1, 8064, 672))
    del arg193_1
    buf146 = buf145; del buf145  # reuse
    buf147 = buf146; del buf146  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_56(c_void_p(buf147.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg318_1
    del arg319_1
    del arg66_1
    del arg67_1
    # Source Nodes: [x_186, x_187], Original ATen: [aten.convolution, aten.silu]
    buf148 = extern_kernels.convolution(buf147, arg194_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf148, (8, 672, 12, 12), (96768, 1, 8064, 672))
    del arg194_1
    del buf147
    buf149 = buf148; del buf148  # reuse
    buf150 = reinterpret_tensor(buf141, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf141  # reuse
    buf151 = reinterpret_tensor(buf150, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf150  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_57(c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()))
    del arg320_1
    del arg321_1
    del arg68_1
    del arg69_1
    # Source Nodes: [x_191, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf152 = extern_kernels.convolution(buf151, arg195_1, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf152, (8, 28, 1, 1), (28, 1, 28, 28))
    del arg195_1
    del arg196_1
    del buf151
    buf153 = buf152; del buf152  # reuse
    cpp_fused_silu_58(c_void_p(buf153.data_ptr()))
    # Source Nodes: [x_se_46, x_se_47], Original ATen: [aten.convolution, aten.silu]
    buf154 = extern_kernels.convolution(buf153, arg197_1, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf154, (8, 672, 1, 1), (672, 1, 672, 672))
    del arg197_1
    del arg198_1
    del buf153
    buf155 = buf149; del buf149  # reuse
    cpp_fused_mul_sigmoid_silu_59(c_void_p(buf155.data_ptr()), c_void_p(buf154.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_191, x_192, x_193], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf156 = extern_kernels.convolution(buf155, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf156, (8, 112, 12, 12), (16128, 1, 1344, 112))
    del arg199_1
    del buf155
    buf157 = buf144; del buf144  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_60(c_void_p(buf157.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg322_1
    del arg323_1
    del arg70_1
    del arg71_1
    del buf156
    # Source Nodes: [x_199], Original ATen: [aten.convolution]
    buf158 = extern_kernels.convolution(buf157, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf158, (8, 672, 12, 12), (96768, 1, 8064, 672))
    del arg200_1
    buf159 = buf158; del buf158  # reuse
    buf160 = buf159; del buf159  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_61(c_void_p(buf160.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()))
    del arg324_1
    del arg325_1
    del arg72_1
    del arg73_1
    # Source Nodes: [x_203, x_204], Original ATen: [aten.convolution, aten.silu]
    buf161 = extern_kernels.convolution(buf160, arg201_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf161, (8, 672, 12, 12), (96768, 1, 8064, 672))
    del arg201_1
    del buf160
    buf162 = buf161; del buf161  # reuse
    buf163 = reinterpret_tensor(buf154, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf154  # reuse
    buf164 = reinterpret_tensor(buf163, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf163  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_62(c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()))
    del arg326_1
    del arg327_1
    del arg74_1
    del arg75_1
    # Source Nodes: [x_208, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf165 = extern_kernels.convolution(buf164, arg202_1, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf165, (8, 28, 1, 1), (28, 1, 28, 28))
    del arg202_1
    del arg203_1
    del buf164
    buf166 = buf165; del buf165  # reuse
    cpp_fused_silu_63(c_void_p(buf166.data_ptr()))
    # Source Nodes: [x_se_50, x_se_51], Original ATen: [aten.convolution, aten.silu]
    buf167 = extern_kernels.convolution(buf166, arg204_1, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf167, (8, 672, 1, 1), (672, 1, 672, 672))
    del arg204_1
    del arg205_1
    del buf166
    buf168 = buf162; del buf162  # reuse
    cpp_fused_mul_sigmoid_silu_64(c_void_p(buf168.data_ptr()), c_void_p(buf167.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_208, x_209, x_210], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf169 = extern_kernels.convolution(buf168, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf169, (8, 112, 12, 12), (16128, 1, 1344, 112))
    del arg206_1
    del buf168
    buf170 = buf157; del buf157  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_65(c_void_p(buf170.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg328_1
    del arg329_1
    del arg76_1
    del arg77_1
    del buf169
    # Source Nodes: [shortcut_13, x_211, x_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf171 = extern_kernels.convolution(buf170, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf171, (8, 672, 12, 12), (96768, 1, 8064, 672))
    del arg207_1
    del buf170
    buf172 = buf171; del buf171  # reuse
    buf173 = buf172; del buf172  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_66(c_void_p(buf173.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()))
    del arg330_1
    del arg331_1
    del arg78_1
    del arg79_1
    # Source Nodes: [x_220, x_221], Original ATen: [aten.convolution, aten.silu]
    buf174 = extern_kernels.convolution(buf173, arg208_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf174, (8, 672, 6, 6), (24192, 1, 4032, 672))
    del arg208_1
    del buf173
    buf175 = buf174; del buf174  # reuse
    buf176 = reinterpret_tensor(buf167, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf167  # reuse
    buf177 = reinterpret_tensor(buf176, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf176  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_67(c_void_p(buf175.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()))
    del arg332_1
    del arg333_1
    del arg80_1
    del arg81_1
    # Source Nodes: [x_225, x_se_52, x_se_53], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf178 = extern_kernels.convolution(buf177, arg209_1, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf178, (8, 28, 1, 1), (28, 1, 28, 28))
    del arg209_1
    del arg210_1
    del buf177
    buf179 = buf178; del buf178  # reuse
    cpp_fused_silu_68(c_void_p(buf179.data_ptr()))
    # Source Nodes: [x_se_54, x_se_55], Original ATen: [aten.convolution, aten.silu]
    buf180 = extern_kernels.convolution(buf179, arg211_1, arg212_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf180, (8, 672, 1, 1), (672, 1, 672, 672))
    del arg211_1
    del arg212_1
    del buf179
    buf181 = buf175; del buf175  # reuse
    cpp_fused_mul_sigmoid_silu_69(c_void_p(buf181.data_ptr()), c_void_p(buf180.data_ptr()))
    del buf180
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_225, x_226, x_227], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf182 = extern_kernels.convolution(buf181, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf182, (8, 192, 6, 6), (6912, 1, 1152, 192))
    del arg213_1
    del buf181
    buf183 = buf182; del buf182  # reuse
    cpp_fused__native_batch_norm_legit_no_training_70(c_void_p(buf183.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg334_1
    del arg335_1
    del arg82_1
    del arg83_1
    # Source Nodes: [x_232], Original ATen: [aten.convolution]
    buf184 = extern_kernels.convolution(buf183, arg214_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf184, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    del arg214_1
    buf185 = buf184; del buf184  # reuse
    buf186 = buf185; del buf185  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_71(c_void_p(buf186.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()))
    del arg336_1
    del arg337_1
    del arg84_1
    del arg85_1
    # Source Nodes: [x_236, x_237], Original ATen: [aten.convolution, aten.silu]
    buf187 = extern_kernels.convolution(buf186, arg215_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
    assert_size_stride(buf187, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    del arg215_1
    del buf186
    buf188 = buf187; del buf187  # reuse
    buf189 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cpu', dtype=torch.float32)
    buf190 = reinterpret_tensor(buf189, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf189  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_72(c_void_p(buf188.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()))
    del arg338_1
    del arg339_1
    del arg86_1
    del arg87_1
    # Source Nodes: [x_241, x_se_56, x_se_57], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf191 = extern_kernels.convolution(buf190, arg216_1, arg217_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf191, (8, 48, 1, 1), (48, 1, 48, 48))
    del arg216_1
    del arg217_1
    del buf190
    buf192 = buf191; del buf191  # reuse
    cpp_fused_silu_73(c_void_p(buf192.data_ptr()))
    # Source Nodes: [x_se_58, x_se_59], Original ATen: [aten.convolution, aten.silu]
    buf193 = extern_kernels.convolution(buf192, arg218_1, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf193, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    del arg218_1
    del arg219_1
    del buf192
    buf194 = buf188; del buf188  # reuse
    cpp_fused_mul_sigmoid_silu_74(c_void_p(buf194.data_ptr()), c_void_p(buf193.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_241, x_242, x_243], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf195 = extern_kernels.convolution(buf194, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf195, (8, 192, 6, 6), (6912, 1, 1152, 192))
    del arg220_1
    del buf194
    buf196 = buf183; del buf183  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_75(c_void_p(buf196.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg340_1
    del arg341_1
    del arg88_1
    del arg89_1
    del buf195
    # Source Nodes: [x_249], Original ATen: [aten.convolution]
    buf197 = extern_kernels.convolution(buf196, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf197, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    del arg221_1
    buf198 = buf197; del buf197  # reuse
    buf199 = buf198; del buf198  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_76(c_void_p(buf199.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()))
    del arg342_1
    del arg343_1
    del arg90_1
    del arg91_1
    # Source Nodes: [x_253, x_254], Original ATen: [aten.convolution, aten.silu]
    buf200 = extern_kernels.convolution(buf199, arg222_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
    assert_size_stride(buf200, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    del arg222_1
    del buf199
    buf201 = buf200; del buf200  # reuse
    buf202 = reinterpret_tensor(buf193, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf193  # reuse
    buf203 = reinterpret_tensor(buf202, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf202  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_77(c_void_p(buf201.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()))
    del arg344_1
    del arg345_1
    del arg92_1
    del arg93_1
    # Source Nodes: [x_258, x_se_60, x_se_61], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf204 = extern_kernels.convolution(buf203, arg223_1, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf204, (8, 48, 1, 1), (48, 1, 48, 48))
    del arg223_1
    del arg224_1
    del buf203
    buf205 = buf204; del buf204  # reuse
    cpp_fused_silu_78(c_void_p(buf205.data_ptr()))
    # Source Nodes: [x_se_62, x_se_63], Original ATen: [aten.convolution, aten.silu]
    buf206 = extern_kernels.convolution(buf205, arg225_1, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf206, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    del arg225_1
    del arg226_1
    del buf205
    buf207 = buf201; del buf201  # reuse
    cpp_fused_mul_sigmoid_silu_79(c_void_p(buf207.data_ptr()), c_void_p(buf206.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_258, x_259, x_260], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf208 = extern_kernels.convolution(buf207, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf208, (8, 192, 6, 6), (6912, 1, 1152, 192))
    del arg227_1
    del buf207
    buf209 = buf196; del buf196  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_80(c_void_p(buf209.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg346_1
    del arg347_1
    del arg94_1
    del arg95_1
    del buf208
    # Source Nodes: [x_266], Original ATen: [aten.convolution]
    buf210 = extern_kernels.convolution(buf209, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf210, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    del arg228_1
    buf211 = buf210; del buf210  # reuse
    buf212 = buf211; del buf211  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_81(c_void_p(buf212.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()))
    del arg348_1
    del arg349_1
    del arg96_1
    del arg97_1
    # Source Nodes: [x_270, x_271], Original ATen: [aten.convolution, aten.silu]
    buf213 = extern_kernels.convolution(buf212, arg229_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
    assert_size_stride(buf213, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    del arg229_1
    del buf212
    buf214 = buf213; del buf213  # reuse
    buf215 = reinterpret_tensor(buf206, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf206  # reuse
    buf216 = reinterpret_tensor(buf215, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf215  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_82(c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()))
    del arg350_1
    del arg351_1
    del arg98_1
    del arg99_1
    # Source Nodes: [x_275, x_se_64, x_se_65], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf217 = extern_kernels.convolution(buf216, arg230_1, arg231_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf217, (8, 48, 1, 1), (48, 1, 48, 48))
    del arg230_1
    del arg231_1
    del buf216
    buf218 = buf217; del buf217  # reuse
    cpp_fused_silu_83(c_void_p(buf218.data_ptr()))
    # Source Nodes: [x_se_66, x_se_67], Original ATen: [aten.convolution, aten.silu]
    buf219 = extern_kernels.convolution(buf218, arg232_1, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf219, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    del arg232_1
    del arg233_1
    del buf218
    buf220 = buf214; del buf214  # reuse
    cpp_fused_mul_sigmoid_silu_84(c_void_p(buf220.data_ptr()), c_void_p(buf219.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_275, x_276, x_277], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf221 = extern_kernels.convolution(buf220, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf221, (8, 192, 6, 6), (6912, 1, 1152, 192))
    del arg234_1
    del buf220
    buf222 = buf209; del buf209  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_85(c_void_p(buf222.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    del arg352_1
    del arg353_1
    del buf221
    # Source Nodes: [x_283], Original ATen: [aten.convolution]
    buf223 = extern_kernels.convolution(buf222, arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf223, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    del arg235_1
    buf224 = buf223; del buf223  # reuse
    buf225 = buf224; del buf224  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_86(c_void_p(buf225.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()))
    del arg102_1
    del arg103_1
    del arg354_1
    del arg355_1
    # Source Nodes: [x_287, x_288], Original ATen: [aten.convolution, aten.silu]
    buf226 = extern_kernels.convolution(buf225, arg236_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
    assert_size_stride(buf226, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    del arg236_1
    del buf225
    buf227 = buf226; del buf226  # reuse
    buf228 = reinterpret_tensor(buf219, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf219  # reuse
    buf229 = reinterpret_tensor(buf228, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf228  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_87(c_void_p(buf227.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()))
    del arg104_1
    del arg105_1
    del arg356_1
    del arg357_1
    # Source Nodes: [x_292, x_se_68, x_se_69], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf230 = extern_kernels.convolution(buf229, arg237_1, arg238_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf230, (8, 48, 1, 1), (48, 1, 48, 48))
    del arg237_1
    del arg238_1
    del buf229
    buf231 = buf230; del buf230  # reuse
    cpp_fused_silu_88(c_void_p(buf231.data_ptr()))
    # Source Nodes: [x_se_70, x_se_71], Original ATen: [aten.convolution, aten.silu]
    buf232 = extern_kernels.convolution(buf231, arg239_1, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf232, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    del arg239_1
    del arg240_1
    del buf231
    buf233 = buf227; del buf227  # reuse
    cpp_fused_mul_sigmoid_silu_89(c_void_p(buf233.data_ptr()), c_void_p(buf232.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate, x_292, x_293, x_294], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf234 = extern_kernels.convolution(buf233, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf234, (8, 192, 6, 6), (6912, 1, 1152, 192))
    del arg241_1
    del buf233
    buf235 = buf222; del buf222  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_90(c_void_p(buf235.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg358_1
    del arg359_1
    del buf234
    # Source Nodes: [shortcut_18, x_295, x_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf236 = extern_kernels.convolution(buf235, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf236, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    del arg242_1
    del buf235
    buf237 = buf236; del buf236  # reuse
    buf238 = buf237; del buf237  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_91(c_void_p(buf238.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()))
    del arg108_1
    del arg109_1
    del arg360_1
    del arg361_1
    # Source Nodes: [x_304, x_305], Original ATen: [aten.convolution, aten.silu]
    buf239 = extern_kernels.convolution(buf238, arg243_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
    assert_size_stride(buf239, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    del arg243_1
    del buf238
    buf240 = buf239; del buf239  # reuse
    buf241 = reinterpret_tensor(buf232, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf232  # reuse
    buf242 = reinterpret_tensor(buf241, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf241  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_92(c_void_p(buf240.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg363_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()))
    del arg110_1
    del arg111_1
    del arg362_1
    del arg363_1
    # Source Nodes: [x_309, x_se_72, x_se_73], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf243 = extern_kernels.convolution(buf242, arg244_1, arg245_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf243, (8, 48, 1, 1), (48, 1, 48, 48))
    del arg244_1
    del arg245_1
    del buf242
    buf244 = buf243; del buf243  # reuse
    cpp_fused_silu_93(c_void_p(buf244.data_ptr()))
    # Source Nodes: [x_se_74, x_se_75], Original ATen: [aten.convolution, aten.silu]
    buf245 = extern_kernels.convolution(buf244, arg246_1, arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf245, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    del arg246_1
    del arg247_1
    del buf244
    buf246 = buf240; del buf240  # reuse
    cpp_fused_mul_sigmoid_silu_94(c_void_p(buf246.data_ptr()), c_void_p(buf245.data_ptr()))
    del buf245
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_309, x_310, x_311], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf247 = extern_kernels.convolution(buf246, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf247, (8, 320, 6, 6), (11520, 1, 1920, 320))
    del arg248_1
    del buf246
    buf248 = buf247; del buf247  # reuse
    cpp_fused__native_batch_norm_legit_no_training_95(c_void_p(buf248.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()))
    del arg112_1
    del arg113_1
    del arg364_1
    del arg365_1
    # Source Nodes: [x_312, x_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf249 = extern_kernels.convolution(buf248, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf249, (8, 1280, 6, 6), (46080, 1, 7680, 1280))
    del arg249_1
    del buf248
    buf250 = buf249; del buf249  # reuse
    buf251 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cpu', dtype=torch.float32)
    buf252 = reinterpret_tensor(buf251, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf251  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_96(c_void_p(buf250.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()))
    del arg114_1
    del arg115_1
    del arg366_1
    del arg367_1
    del buf250
    buf253 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_326], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg251_1, reinterpret_tensor(buf252, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg250_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf253)
    del arg250_1
    del arg251_1
    return (buf253, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((4, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((6, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((6, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((28, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((28, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((28, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((28, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((8, 3, 192, 192), (110592, 36864, 192, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tinynet_a', benchmark_compiled_module)
