
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
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
                    auto tmp4 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x2) + (56448L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(72L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (72L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                        auto tmp1 = static_cast<float>(3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp12 = tmp10 * tmp11;
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x2) + (94080L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp1 = static_cast<float>(3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp12 = tmp10 * tmp11;
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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
                    auto tmp4 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x2) + (94080L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp1 = static_cast<float>(3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp12 = tmp10 * tmp11;
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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
                    auto tmp4 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(752640L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(188160L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(156800L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(156800L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_30 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
                auto tmp4 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144256L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144256L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_33 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
                auto tmp4 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144256L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144256L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_36 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
                auto tmp4 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376320L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                            auto tmp1 = static_cast<float>(3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = at::vec::maximum(tmp3, tmp5);
                            auto tmp7 = static_cast<float>(6.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = at::vec::minimum(tmp6, tmp8);
                            auto tmp10 = tmp0 * tmp9;
                            auto tmp11 = tmp10 / tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
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
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_hardswish_mul_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp1 = static_cast<float>(3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp12 = tmp11 + tmp2;
                        auto tmp13 = at::vec::maximum(tmp12, tmp5);
                        auto tmp14 = at::vec::minimum(tmp13, tmp8);
                        auto tmp15 = tmp11 * tmp14;
                        auto tmp16 = tmp15 / tmp8;
                        auto tmp17 = tmp10 * tmp16;
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(0.001);
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
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(526848L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x2) + (131712L*x0)));
                            auto tmp1 = static_cast<float>(3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = at::vec::maximum(tmp3, tmp5);
                            auto tmp7 = static_cast<float>(6.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = at::vec::minimum(tmp6, tmp8);
                            auto tmp10 = tmp0 * tmp9;
                            auto tmp11 = tmp10 / tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_hardswish_mul_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp1 = static_cast<float>(3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp12 = tmp11 + tmp2;
                        auto tmp13 = at::vec::maximum(tmp12, tmp5);
                        auto tmp14 = at::vec::minimum(tmp13, tmp8);
                        auto tmp15 = tmp11 * tmp14;
                        auto tmp16 = tmp15 / tmp8;
                        auto tmp17 = tmp10 * tmp16;
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_46 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
                auto tmp4 = static_cast<float>(0.001);
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
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(526848L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x2) + (32928L*x0)));
                            auto tmp1 = static_cast<float>(3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = at::vec::maximum(tmp3, tmp5);
                            auto tmp7 = static_cast<float>(6.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = at::vec::minimum(tmp6, tmp8);
                            auto tmp10 = tmp0 * tmp9;
                            auto tmp11 = tmp10 / tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_hardswish_mul_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp1 = static_cast<float>(3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp12 = tmp11 + tmp2;
                        auto tmp13 = at::vec::maximum(tmp12, tmp5);
                        auto tmp14 = at::vec::minimum(tmp13, tmp8);
                        auto tmp15 = tmp11 * tmp14;
                        auto tmp16 = tmp15 / tmp8;
                        auto tmp17 = tmp10 * tmp16;
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(0.001);
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(188160L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            auto tmp1 = static_cast<float>(3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = at::vec::maximum(tmp3, tmp5);
                            auto tmp7 = static_cast<float>(6.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = at::vec::minimum(tmp6, tmp8);
                            auto tmp10 = tmp0 * tmp9;
                            auto tmp11 = tmp10 / tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_hardswish_mul_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp1 = static_cast<float>(3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp12 = tmp11 + tmp2;
                        auto tmp13 = at::vec::maximum(tmp12, tmp5);
                        auto tmp14 = at::vec::minimum(tmp13, tmp8);
                        auto tmp15 = tmp11 * tmp14;
                        auto tmp16 = tmp15 / tmp8;
                        auto tmp17 = tmp10 * tmp16;
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_56 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(0.001);
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(188160L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            auto tmp1 = static_cast<float>(3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = at::vec::maximum(tmp3, tmp5);
                            auto tmp7 = static_cast<float>(6.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = at::vec::minimum(tmp6, tmp8);
                            auto tmp10 = tmp0 * tmp9;
                            auto tmp11 = tmp10 / tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_hardswish_mul_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp1 = static_cast<float>(3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp12 = tmp11 + tmp2;
                        auto tmp13 = at::vec::maximum(tmp12, tmp5);
                        auto tmp14 = at::vec::minimum(tmp13, tmp8);
                        auto tmp15 = tmp11 * tmp14;
                        auto tmp16 = tmp15 / tmp8;
                        auto tmp17 = tmp10 * tmp16;
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_61 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(0.001);
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            auto tmp1 = static_cast<float>(3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = at::vec::maximum(tmp3, tmp5);
                            auto tmp7 = static_cast<float>(6.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = at::vec::minimum(tmp6, tmp8);
                            auto tmp10 = tmp0 * tmp9;
                            auto tmp11 = tmp10 / tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(5120L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (16, ), (1, ))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (16, ), (1, ))
    assert_size_stride(arg9_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg16_1, (24, ), (1, ))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg19_1, (72, ), (1, ))
    assert_size_stride(arg20_1, (72, ), (1, ))
    assert_size_stride(arg21_1, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg22_1, (72, ), (1, ))
    assert_size_stride(arg23_1, (72, ), (1, ))
    assert_size_stride(arg24_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg25_1, (24, ), (1, ))
    assert_size_stride(arg26_1, (24, ), (1, ))
    assert_size_stride(arg27_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg28_1, (72, ), (1, ))
    assert_size_stride(arg29_1, (72, ), (1, ))
    assert_size_stride(arg30_1, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg31_1, (72, ), (1, ))
    assert_size_stride(arg32_1, (72, ), (1, ))
    assert_size_stride(arg33_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg34_1, (24, ), (1, ))
    assert_size_stride(arg35_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg36_1, (72, ), (1, ))
    assert_size_stride(arg37_1, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg38_1, (40, ), (1, ))
    assert_size_stride(arg39_1, (40, ), (1, ))
    assert_size_stride(arg40_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg41_1, (120, ), (1, ))
    assert_size_stride(arg42_1, (120, ), (1, ))
    assert_size_stride(arg43_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg44_1, (120, ), (1, ))
    assert_size_stride(arg45_1, (120, ), (1, ))
    assert_size_stride(arg46_1, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg47_1, (32, ), (1, ))
    assert_size_stride(arg48_1, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg49_1, (120, ), (1, ))
    assert_size_stride(arg50_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg51_1, (40, ), (1, ))
    assert_size_stride(arg52_1, (40, ), (1, ))
    assert_size_stride(arg53_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg54_1, (120, ), (1, ))
    assert_size_stride(arg55_1, (120, ), (1, ))
    assert_size_stride(arg56_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg57_1, (120, ), (1, ))
    assert_size_stride(arg58_1, (120, ), (1, ))
    assert_size_stride(arg59_1, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg60_1, (32, ), (1, ))
    assert_size_stride(arg61_1, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg62_1, (120, ), (1, ))
    assert_size_stride(arg63_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg64_1, (40, ), (1, ))
    assert_size_stride(arg65_1, (40, ), (1, ))
    assert_size_stride(arg66_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg67_1, (240, ), (1, ))
    assert_size_stride(arg68_1, (240, ), (1, ))
    assert_size_stride(arg69_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg70_1, (240, ), (1, ))
    assert_size_stride(arg71_1, (240, ), (1, ))
    assert_size_stride(arg72_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg73_1, (80, ), (1, ))
    assert_size_stride(arg74_1, (80, ), (1, ))
    assert_size_stride(arg75_1, (200, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg76_1, (200, ), (1, ))
    assert_size_stride(arg77_1, (200, ), (1, ))
    assert_size_stride(arg78_1, (200, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg79_1, (200, ), (1, ))
    assert_size_stride(arg80_1, (200, ), (1, ))
    assert_size_stride(arg81_1, (80, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg82_1, (80, ), (1, ))
    assert_size_stride(arg83_1, (80, ), (1, ))
    assert_size_stride(arg84_1, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg85_1, (184, ), (1, ))
    assert_size_stride(arg86_1, (184, ), (1, ))
    assert_size_stride(arg87_1, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg88_1, (184, ), (1, ))
    assert_size_stride(arg89_1, (184, ), (1, ))
    assert_size_stride(arg90_1, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg91_1, (80, ), (1, ))
    assert_size_stride(arg92_1, (80, ), (1, ))
    assert_size_stride(arg93_1, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg94_1, (184, ), (1, ))
    assert_size_stride(arg95_1, (184, ), (1, ))
    assert_size_stride(arg96_1, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg97_1, (184, ), (1, ))
    assert_size_stride(arg98_1, (184, ), (1, ))
    assert_size_stride(arg99_1, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg100_1, (80, ), (1, ))
    assert_size_stride(arg101_1, (80, ), (1, ))
    assert_size_stride(arg102_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg103_1, (480, ), (1, ))
    assert_size_stride(arg104_1, (480, ), (1, ))
    assert_size_stride(arg105_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg106_1, (480, ), (1, ))
    assert_size_stride(arg107_1, (480, ), (1, ))
    assert_size_stride(arg108_1, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg109_1, (120, ), (1, ))
    assert_size_stride(arg110_1, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg111_1, (480, ), (1, ))
    assert_size_stride(arg112_1, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg113_1, (112, ), (1, ))
    assert_size_stride(arg114_1, (112, ), (1, ))
    assert_size_stride(arg115_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg116_1, (672, ), (1, ))
    assert_size_stride(arg117_1, (672, ), (1, ))
    assert_size_stride(arg118_1, (672, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg119_1, (672, ), (1, ))
    assert_size_stride(arg120_1, (672, ), (1, ))
    assert_size_stride(arg121_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg122_1, (168, ), (1, ))
    assert_size_stride(arg123_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg124_1, (672, ), (1, ))
    assert_size_stride(arg125_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg126_1, (112, ), (1, ))
    assert_size_stride(arg127_1, (112, ), (1, ))
    assert_size_stride(arg128_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg129_1, (672, ), (1, ))
    assert_size_stride(arg130_1, (672, ), (1, ))
    assert_size_stride(arg131_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg132_1, (672, ), (1, ))
    assert_size_stride(arg133_1, (672, ), (1, ))
    assert_size_stride(arg134_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg135_1, (168, ), (1, ))
    assert_size_stride(arg136_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg137_1, (672, ), (1, ))
    assert_size_stride(arg138_1, (160, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg139_1, (160, ), (1, ))
    assert_size_stride(arg140_1, (160, ), (1, ))
    assert_size_stride(arg141_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg142_1, (960, ), (1, ))
    assert_size_stride(arg143_1, (960, ), (1, ))
    assert_size_stride(arg144_1, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg145_1, (960, ), (1, ))
    assert_size_stride(arg146_1, (960, ), (1, ))
    assert_size_stride(arg147_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg148_1, (240, ), (1, ))
    assert_size_stride(arg149_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg150_1, (960, ), (1, ))
    assert_size_stride(arg151_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg152_1, (160, ), (1, ))
    assert_size_stride(arg153_1, (160, ), (1, ))
    assert_size_stride(arg154_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg155_1, (960, ), (1, ))
    assert_size_stride(arg156_1, (960, ), (1, ))
    assert_size_stride(arg157_1, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg158_1, (960, ), (1, ))
    assert_size_stride(arg159_1, (960, ), (1, ))
    assert_size_stride(arg160_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg161_1, (240, ), (1, ))
    assert_size_stride(arg162_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg163_1, (960, ), (1, ))
    assert_size_stride(arg164_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg165_1, (160, ), (1, ))
    assert_size_stride(arg166_1, (160, ), (1, ))
    assert_size_stride(arg167_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg168_1, (960, ), (1, ))
    assert_size_stride(arg169_1, (960, ), (1, ))
    assert_size_stride(arg170_1, (1280, 960), (960, 1))
    assert_size_stride(arg171_1, (1280, ), (1, ))
    assert_size_stride(arg172_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg173_1, (1000, ), (1, ))
    assert_size_stride(arg174_1, (16, ), (1, ))
    assert_size_stride(arg175_1, (16, ), (1, ))
    assert_size_stride(arg176_1, (), ())
    assert_size_stride(arg177_1, (16, ), (1, ))
    assert_size_stride(arg178_1, (16, ), (1, ))
    assert_size_stride(arg179_1, (), ())
    assert_size_stride(arg180_1, (16, ), (1, ))
    assert_size_stride(arg181_1, (16, ), (1, ))
    assert_size_stride(arg182_1, (), ())
    assert_size_stride(arg183_1, (64, ), (1, ))
    assert_size_stride(arg184_1, (64, ), (1, ))
    assert_size_stride(arg185_1, (), ())
    assert_size_stride(arg186_1, (64, ), (1, ))
    assert_size_stride(arg187_1, (64, ), (1, ))
    assert_size_stride(arg188_1, (), ())
    assert_size_stride(arg189_1, (24, ), (1, ))
    assert_size_stride(arg190_1, (24, ), (1, ))
    assert_size_stride(arg191_1, (), ())
    assert_size_stride(arg192_1, (72, ), (1, ))
    assert_size_stride(arg193_1, (72, ), (1, ))
    assert_size_stride(arg194_1, (), ())
    assert_size_stride(arg195_1, (72, ), (1, ))
    assert_size_stride(arg196_1, (72, ), (1, ))
    assert_size_stride(arg197_1, (), ())
    assert_size_stride(arg198_1, (24, ), (1, ))
    assert_size_stride(arg199_1, (24, ), (1, ))
    assert_size_stride(arg200_1, (), ())
    assert_size_stride(arg201_1, (72, ), (1, ))
    assert_size_stride(arg202_1, (72, ), (1, ))
    assert_size_stride(arg203_1, (), ())
    assert_size_stride(arg204_1, (72, ), (1, ))
    assert_size_stride(arg205_1, (72, ), (1, ))
    assert_size_stride(arg206_1, (), ())
    assert_size_stride(arg207_1, (40, ), (1, ))
    assert_size_stride(arg208_1, (40, ), (1, ))
    assert_size_stride(arg209_1, (), ())
    assert_size_stride(arg210_1, (120, ), (1, ))
    assert_size_stride(arg211_1, (120, ), (1, ))
    assert_size_stride(arg212_1, (), ())
    assert_size_stride(arg213_1, (120, ), (1, ))
    assert_size_stride(arg214_1, (120, ), (1, ))
    assert_size_stride(arg215_1, (), ())
    assert_size_stride(arg216_1, (40, ), (1, ))
    assert_size_stride(arg217_1, (40, ), (1, ))
    assert_size_stride(arg218_1, (), ())
    assert_size_stride(arg219_1, (120, ), (1, ))
    assert_size_stride(arg220_1, (120, ), (1, ))
    assert_size_stride(arg221_1, (), ())
    assert_size_stride(arg222_1, (120, ), (1, ))
    assert_size_stride(arg223_1, (120, ), (1, ))
    assert_size_stride(arg224_1, (), ())
    assert_size_stride(arg225_1, (40, ), (1, ))
    assert_size_stride(arg226_1, (40, ), (1, ))
    assert_size_stride(arg227_1, (), ())
    assert_size_stride(arg228_1, (240, ), (1, ))
    assert_size_stride(arg229_1, (240, ), (1, ))
    assert_size_stride(arg230_1, (), ())
    assert_size_stride(arg231_1, (240, ), (1, ))
    assert_size_stride(arg232_1, (240, ), (1, ))
    assert_size_stride(arg233_1, (), ())
    assert_size_stride(arg234_1, (80, ), (1, ))
    assert_size_stride(arg235_1, (80, ), (1, ))
    assert_size_stride(arg236_1, (), ())
    assert_size_stride(arg237_1, (200, ), (1, ))
    assert_size_stride(arg238_1, (200, ), (1, ))
    assert_size_stride(arg239_1, (), ())
    assert_size_stride(arg240_1, (200, ), (1, ))
    assert_size_stride(arg241_1, (200, ), (1, ))
    assert_size_stride(arg242_1, (), ())
    assert_size_stride(arg243_1, (80, ), (1, ))
    assert_size_stride(arg244_1, (80, ), (1, ))
    assert_size_stride(arg245_1, (), ())
    assert_size_stride(arg246_1, (184, ), (1, ))
    assert_size_stride(arg247_1, (184, ), (1, ))
    assert_size_stride(arg248_1, (), ())
    assert_size_stride(arg249_1, (184, ), (1, ))
    assert_size_stride(arg250_1, (184, ), (1, ))
    assert_size_stride(arg251_1, (), ())
    assert_size_stride(arg252_1, (80, ), (1, ))
    assert_size_stride(arg253_1, (80, ), (1, ))
    assert_size_stride(arg254_1, (), ())
    assert_size_stride(arg255_1, (184, ), (1, ))
    assert_size_stride(arg256_1, (184, ), (1, ))
    assert_size_stride(arg257_1, (), ())
    assert_size_stride(arg258_1, (184, ), (1, ))
    assert_size_stride(arg259_1, (184, ), (1, ))
    assert_size_stride(arg260_1, (), ())
    assert_size_stride(arg261_1, (80, ), (1, ))
    assert_size_stride(arg262_1, (80, ), (1, ))
    assert_size_stride(arg263_1, (), ())
    assert_size_stride(arg264_1, (480, ), (1, ))
    assert_size_stride(arg265_1, (480, ), (1, ))
    assert_size_stride(arg266_1, (), ())
    assert_size_stride(arg267_1, (480, ), (1, ))
    assert_size_stride(arg268_1, (480, ), (1, ))
    assert_size_stride(arg269_1, (), ())
    assert_size_stride(arg270_1, (112, ), (1, ))
    assert_size_stride(arg271_1, (112, ), (1, ))
    assert_size_stride(arg272_1, (), ())
    assert_size_stride(arg273_1, (672, ), (1, ))
    assert_size_stride(arg274_1, (672, ), (1, ))
    assert_size_stride(arg275_1, (), ())
    assert_size_stride(arg276_1, (672, ), (1, ))
    assert_size_stride(arg277_1, (672, ), (1, ))
    assert_size_stride(arg278_1, (), ())
    assert_size_stride(arg279_1, (112, ), (1, ))
    assert_size_stride(arg280_1, (112, ), (1, ))
    assert_size_stride(arg281_1, (), ())
    assert_size_stride(arg282_1, (672, ), (1, ))
    assert_size_stride(arg283_1, (672, ), (1, ))
    assert_size_stride(arg284_1, (), ())
    assert_size_stride(arg285_1, (672, ), (1, ))
    assert_size_stride(arg286_1, (672, ), (1, ))
    assert_size_stride(arg287_1, (), ())
    assert_size_stride(arg288_1, (160, ), (1, ))
    assert_size_stride(arg289_1, (160, ), (1, ))
    assert_size_stride(arg290_1, (), ())
    assert_size_stride(arg291_1, (960, ), (1, ))
    assert_size_stride(arg292_1, (960, ), (1, ))
    assert_size_stride(arg293_1, (), ())
    assert_size_stride(arg294_1, (960, ), (1, ))
    assert_size_stride(arg295_1, (960, ), (1, ))
    assert_size_stride(arg296_1, (), ())
    assert_size_stride(arg297_1, (160, ), (1, ))
    assert_size_stride(arg298_1, (160, ), (1, ))
    assert_size_stride(arg299_1, (), ())
    assert_size_stride(arg300_1, (960, ), (1, ))
    assert_size_stride(arg301_1, (960, ), (1, ))
    assert_size_stride(arg302_1, (), ())
    assert_size_stride(arg303_1, (960, ), (1, ))
    assert_size_stride(arg304_1, (960, ), (1, ))
    assert_size_stride(arg305_1, (), ())
    assert_size_stride(arg306_1, (160, ), (1, ))
    assert_size_stride(arg307_1, (160, ), (1, ))
    assert_size_stride(arg308_1, (), ())
    assert_size_stride(arg309_1, (960, ), (1, ))
    assert_size_stride(arg310_1, (960, ), (1, ))
    assert_size_stride(arg311_1, (), ())
    assert_size_stride(arg312_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg312_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg312_1
    # Source Nodes: [l__mod___features_0_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (4, 16, 112, 112), (200704, 1, 1792, 16))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = buf3; del buf3  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_1(c_void_p(buf4.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()))
    del arg174_1
    del arg175_1
    del arg1_1
    del arg2_1
    # Source Nodes: [getattr_l__mod___features___1___block_0_0, l__mod___features_0_2], Original ATen: [aten.convolution, aten.hardswish]
    buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
    assert_size_stride(buf5, (4, 16, 112, 112), (200704, 1, 1792, 16))
    del arg3_1
    buf6 = buf5; del buf5  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_2(c_void_p(buf6.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg177_1
    del arg178_1
    del arg4_1
    del arg5_1
    # Source Nodes: [getattr_l__mod___features___1___block_0_1, getattr_l__mod___features___1___block_0_2, getattr_l__mod___features___1___block_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf7 = extern_kernels.convolution(buf6, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf7, (4, 16, 112, 112), (200704, 1, 1792, 16))
    del arg6_1
    del buf6
    buf8 = buf4; del buf4  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_3(c_void_p(buf8.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()))
    del arg180_1
    del arg181_1
    del arg7_1
    del arg8_1
    del buf7
    # Source Nodes: [getattr_l__mod___features___2___block_0_0, result, result_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf9 = extern_kernels.convolution(buf8, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf9, (4, 64, 112, 112), (802816, 1, 7168, 64))
    del arg9_1
    del buf8
    buf10 = buf9; del buf9  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_4(c_void_p(buf10.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()))
    del arg10_1
    del arg11_1
    del arg183_1
    del arg184_1
    # Source Nodes: [getattr_l__mod___features___2___block_0_1, getattr_l__mod___features___2___block_0_2, getattr_l__mod___features___2___block_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf11 = extern_kernels.convolution(buf10, arg12_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf11, (4, 64, 56, 56), (200704, 1, 3584, 64))
    del arg12_1
    del buf10
    buf12 = buf11; del buf11  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_5(c_void_p(buf12.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()))
    del arg13_1
    del arg14_1
    del arg186_1
    del arg187_1
    # Source Nodes: [getattr_l__mod___features___2___block_1_1, getattr_l__mod___features___2___block_1_2, getattr_l__mod___features___2___block_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf13 = extern_kernels.convolution(buf12, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf13, (4, 24, 56, 56), (75264, 1, 1344, 24))
    del arg15_1
    del buf12
    buf14 = buf13; del buf13  # reuse
    cpp_fused__native_batch_norm_legit_no_training_6(c_void_p(buf14.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg16_1
    del arg17_1
    del arg189_1
    del arg190_1
    # Source Nodes: [getattr_l__mod___features___3___block_0_0], Original ATen: [aten.convolution]
    buf15 = extern_kernels.convolution(buf14, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf15, (4, 72, 56, 56), (225792, 1, 4032, 72))
    del arg18_1
    buf16 = buf15; del buf15  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_7(c_void_p(buf16.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()))
    del arg192_1
    del arg193_1
    del arg19_1
    del arg20_1
    # Source Nodes: [getattr_l__mod___features___3___block_0_1, getattr_l__mod___features___3___block_0_2, getattr_l__mod___features___3___block_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf17 = extern_kernels.convolution(buf16, arg21_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
    assert_size_stride(buf17, (4, 72, 56, 56), (225792, 1, 4032, 72))
    del arg21_1
    del buf16
    buf18 = buf17; del buf17  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_8(c_void_p(buf18.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg195_1
    del arg196_1
    del arg22_1
    del arg23_1
    # Source Nodes: [getattr_l__mod___features___3___block_1_1, getattr_l__mod___features___3___block_1_2, getattr_l__mod___features___3___block_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf19 = extern_kernels.convolution(buf18, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf19, (4, 24, 56, 56), (75264, 1, 1344, 24))
    del arg24_1
    del buf18
    buf20 = buf14; del buf14  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_9(c_void_p(buf20.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()))
    del arg198_1
    del arg199_1
    del arg25_1
    del arg26_1
    del buf19
    # Source Nodes: [getattr_l__mod___features___4___block_0_0, result_3, result_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf21 = extern_kernels.convolution(buf20, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf21, (4, 72, 56, 56), (225792, 1, 4032, 72))
    del arg27_1
    del buf20
    buf22 = buf21; del buf21  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_10(c_void_p(buf22.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg201_1
    del arg202_1
    del arg28_1
    del arg29_1
    # Source Nodes: [getattr_l__mod___features___4___block_0_1, getattr_l__mod___features___4___block_0_2, getattr_l__mod___features___4___block_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf23 = extern_kernels.convolution(buf22, arg30_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
    assert_size_stride(buf23, (4, 72, 28, 28), (56448, 1, 2016, 72))
    del arg30_1
    del buf22
    buf24 = buf23; del buf23  # reuse
    buf25 = empty_strided((4, 72, 1, 1), (72, 1, 288, 288), device='cpu', dtype=torch.float32)
    buf26 = reinterpret_tensor(buf25, (4, 72, 1, 1), (72, 1, 72, 72), 0); del buf25  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_11(c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()))
    del arg204_1
    del arg205_1
    del arg31_1
    del arg32_1
    # Source Nodes: [scale, scale_1], Original ATen: [aten.convolution, aten.mean]
    buf27 = extern_kernels.convolution(buf26, arg33_1, arg34_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf27, (4, 24, 1, 1), (24, 1, 24, 24))
    del arg33_1
    del arg34_1
    del buf26
    buf28 = buf27; del buf27  # reuse
    cpp_fused_relu_12(c_void_p(buf28.data_ptr()))
    # Source Nodes: [scale_2, scale_3], Original ATen: [aten.convolution, aten.relu]
    buf29 = extern_kernels.convolution(buf28, arg35_1, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf29, (4, 72, 1, 1), (72, 1, 72, 72))
    del arg35_1
    del arg36_1
    del buf28
    buf30 = buf24; del buf24  # reuse
    cpp_fused_hardsigmoid_mul_13(c_void_p(buf30.data_ptr()), c_void_p(buf29.data_ptr()))
    del buf29
    # Source Nodes: [getattr_l__mod___features___4___block_3_0, mul, scale_4], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mul]
    buf31 = extern_kernels.convolution(buf30, arg37_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf31, (4, 40, 28, 28), (31360, 1, 1120, 40))
    del arg37_1
    del buf30
    buf32 = buf31; del buf31  # reuse
    cpp_fused__native_batch_norm_legit_no_training_14(c_void_p(buf32.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()))
    del arg207_1
    del arg208_1
    del arg38_1
    del arg39_1
    # Source Nodes: [getattr_l__mod___features___5___block_0_0], Original ATen: [aten.convolution]
    buf33 = extern_kernels.convolution(buf32, arg40_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf33, (4, 120, 28, 28), (94080, 1, 3360, 120))
    del arg40_1
    buf34 = buf33; del buf33  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_15(c_void_p(buf34.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(arg42_1.data_ptr()))
    del arg210_1
    del arg211_1
    del arg41_1
    del arg42_1
    # Source Nodes: [getattr_l__mod___features___5___block_0_1, getattr_l__mod___features___5___block_0_2, getattr_l__mod___features___5___block_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf35 = extern_kernels.convolution(buf34, arg43_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf35, (4, 120, 28, 28), (94080, 1, 3360, 120))
    del arg43_1
    del buf34
    buf36 = buf35; del buf35  # reuse
    buf37 = empty_strided((4, 120, 1, 1), (120, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf38 = reinterpret_tensor(buf37, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf37  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_16(c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()))
    del arg213_1
    del arg214_1
    del arg44_1
    del arg45_1
    # Source Nodes: [scale_5, scale_6], Original ATen: [aten.convolution, aten.mean]
    buf39 = extern_kernels.convolution(buf38, arg46_1, arg47_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf39, (4, 32, 1, 1), (32, 1, 32, 32))
    del arg46_1
    del arg47_1
    del buf38
    buf40 = buf39; del buf39  # reuse
    cpp_fused_relu_17(c_void_p(buf40.data_ptr()))
    # Source Nodes: [scale_7, scale_8], Original ATen: [aten.convolution, aten.relu]
    buf41 = extern_kernels.convolution(buf40, arg48_1, arg49_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf41, (4, 120, 1, 1), (120, 1, 120, 120))
    del arg48_1
    del arg49_1
    del buf40
    buf42 = buf36; del buf36  # reuse
    cpp_fused_hardsigmoid_mul_18(c_void_p(buf42.data_ptr()), c_void_p(buf41.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___5___block_3_0, mul_1, scale_9], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mul]
    buf43 = extern_kernels.convolution(buf42, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf43, (4, 40, 28, 28), (31360, 1, 1120, 40))
    del arg50_1
    del buf42
    buf44 = buf32; del buf32  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_19(c_void_p(buf44.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg52_1.data_ptr()))
    del arg216_1
    del arg217_1
    del arg51_1
    del arg52_1
    del buf43
    # Source Nodes: [getattr_l__mod___features___6___block_0_0], Original ATen: [aten.convolution]
    buf45 = extern_kernels.convolution(buf44, arg53_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf45, (4, 120, 28, 28), (94080, 1, 3360, 120))
    del arg53_1
    buf46 = buf45; del buf45  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_20(c_void_p(buf46.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()))
    del arg219_1
    del arg220_1
    del arg54_1
    del arg55_1
    # Source Nodes: [getattr_l__mod___features___6___block_0_1, getattr_l__mod___features___6___block_0_2, getattr_l__mod___features___6___block_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf47 = extern_kernels.convolution(buf46, arg56_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf47, (4, 120, 28, 28), (94080, 1, 3360, 120))
    del arg56_1
    del buf46
    buf48 = buf47; del buf47  # reuse
    buf49 = reinterpret_tensor(buf41, (4, 120, 1, 1), (120, 1, 480, 480), 0); del buf41  # reuse
    buf50 = reinterpret_tensor(buf49, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf49  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_21(c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(arg58_1.data_ptr()))
    del arg222_1
    del arg223_1
    del arg57_1
    del arg58_1
    # Source Nodes: [scale_10, scale_11], Original ATen: [aten.convolution, aten.mean]
    buf51 = extern_kernels.convolution(buf50, arg59_1, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf51, (4, 32, 1, 1), (32, 1, 32, 32))
    del arg59_1
    del arg60_1
    del buf50
    buf52 = buf51; del buf51  # reuse
    cpp_fused_relu_22(c_void_p(buf52.data_ptr()))
    # Source Nodes: [scale_12, scale_13], Original ATen: [aten.convolution, aten.relu]
    buf53 = extern_kernels.convolution(buf52, arg61_1, arg62_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf53, (4, 120, 1, 1), (120, 1, 120, 120))
    del arg61_1
    del arg62_1
    del buf52
    buf54 = buf48; del buf48  # reuse
    cpp_fused_hardsigmoid_mul_23(c_void_p(buf54.data_ptr()), c_void_p(buf53.data_ptr()))
    del buf53
    # Source Nodes: [getattr_l__mod___features___6___block_3_0, mul_2, scale_14], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mul]
    buf55 = extern_kernels.convolution(buf54, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf55, (4, 40, 28, 28), (31360, 1, 1120, 40))
    del arg63_1
    del buf54
    buf56 = buf44; del buf44  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_24(c_void_p(buf56.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg225_1
    del arg226_1
    del arg64_1
    del arg65_1
    del buf55
    # Source Nodes: [getattr_l__mod___features___7___block_0_0, result_8, result_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf57 = extern_kernels.convolution(buf56, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf57, (4, 240, 28, 28), (188160, 1, 6720, 240))
    del arg66_1
    del buf56
    buf58 = buf57; del buf57  # reuse
    buf59 = buf58; del buf58  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_25(c_void_p(buf59.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()))
    del arg228_1
    del arg229_1
    del arg67_1
    del arg68_1
    # Source Nodes: [getattr_l__mod___features___7___block_0_2, getattr_l__mod___features___7___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
    buf60 = extern_kernels.convolution(buf59, arg69_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf60, (4, 240, 14, 14), (47040, 1, 3360, 240))
    del arg69_1
    del buf59
    buf61 = buf60; del buf60  # reuse
    buf62 = buf61; del buf61  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_26(c_void_p(buf62.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg231_1
    del arg232_1
    del arg70_1
    del arg71_1
    # Source Nodes: [getattr_l__mod___features___7___block_1_2, getattr_l__mod___features___7___block_2_0], Original ATen: [aten.convolution, aten.hardswish]
    buf63 = extern_kernels.convolution(buf62, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf63, (4, 80, 14, 14), (15680, 1, 1120, 80))
    del arg72_1
    del buf62
    buf64 = buf63; del buf63  # reuse
    cpp_fused__native_batch_norm_legit_no_training_27(c_void_p(buf64.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()))
    del arg234_1
    del arg235_1
    del arg73_1
    del arg74_1
    # Source Nodes: [getattr_l__mod___features___8___block_0_0], Original ATen: [aten.convolution]
    buf65 = extern_kernels.convolution(buf64, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf65, (4, 200, 14, 14), (39200, 1, 2800, 200))
    del arg75_1
    buf66 = buf65; del buf65  # reuse
    buf67 = buf66; del buf66  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_28(c_void_p(buf67.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg237_1
    del arg238_1
    del arg76_1
    del arg77_1
    # Source Nodes: [getattr_l__mod___features___8___block_0_2, getattr_l__mod___features___8___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
    buf68 = extern_kernels.convolution(buf67, arg78_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=200, bias=None)
    assert_size_stride(buf68, (4, 200, 14, 14), (39200, 1, 2800, 200))
    del arg78_1
    del buf67
    buf69 = buf68; del buf68  # reuse
    buf70 = buf69; del buf69  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_29(c_void_p(buf70.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()))
    del arg240_1
    del arg241_1
    del arg79_1
    del arg80_1
    # Source Nodes: [getattr_l__mod___features___8___block_1_2, getattr_l__mod___features___8___block_2_0], Original ATen: [aten.convolution, aten.hardswish]
    buf71 = extern_kernels.convolution(buf70, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf71, (4, 80, 14, 14), (15680, 1, 1120, 80))
    del arg81_1
    del buf70
    buf72 = buf64; del buf64  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_30(c_void_p(buf72.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg243_1
    del arg244_1
    del arg82_1
    del arg83_1
    del buf71
    # Source Nodes: [getattr_l__mod___features___9___block_0_0], Original ATen: [aten.convolution]
    buf73 = extern_kernels.convolution(buf72, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf73, (4, 184, 14, 14), (36064, 1, 2576, 184))
    del arg84_1
    buf74 = buf73; del buf73  # reuse
    buf75 = buf74; del buf74  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_31(c_void_p(buf75.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(arg86_1.data_ptr()))
    del arg246_1
    del arg247_1
    del arg85_1
    del arg86_1
    # Source Nodes: [getattr_l__mod___features___9___block_0_2, getattr_l__mod___features___9___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
    buf76 = extern_kernels.convolution(buf75, arg87_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
    assert_size_stride(buf76, (4, 184, 14, 14), (36064, 1, 2576, 184))
    del arg87_1
    del buf75
    buf77 = buf76; del buf76  # reuse
    buf78 = buf77; del buf77  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_32(c_void_p(buf78.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg249_1
    del arg250_1
    del arg88_1
    del arg89_1
    # Source Nodes: [getattr_l__mod___features___9___block_1_2, getattr_l__mod___features___9___block_2_0], Original ATen: [aten.convolution, aten.hardswish]
    buf79 = extern_kernels.convolution(buf78, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf79, (4, 80, 14, 14), (15680, 1, 1120, 80))
    del arg90_1
    del buf78
    buf80 = buf72; del buf72  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_33(c_void_p(buf80.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg92_1.data_ptr()))
    del arg252_1
    del arg253_1
    del arg91_1
    del arg92_1
    del buf79
    # Source Nodes: [getattr_l__mod___features___10___block_0_0], Original ATen: [aten.convolution]
    buf81 = extern_kernels.convolution(buf80, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf81, (4, 184, 14, 14), (36064, 1, 2576, 184))
    del arg93_1
    buf82 = buf81; del buf81  # reuse
    buf83 = buf82; del buf82  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_34(c_void_p(buf83.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg255_1
    del arg256_1
    del arg94_1
    del arg95_1
    # Source Nodes: [getattr_l__mod___features___10___block_0_2, getattr_l__mod___features___10___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
    buf84 = extern_kernels.convolution(buf83, arg96_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
    assert_size_stride(buf84, (4, 184, 14, 14), (36064, 1, 2576, 184))
    del arg96_1
    del buf83
    buf85 = buf84; del buf84  # reuse
    buf86 = buf85; del buf85  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_35(c_void_p(buf86.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg98_1.data_ptr()))
    del arg258_1
    del arg259_1
    del arg97_1
    del arg98_1
    # Source Nodes: [getattr_l__mod___features___10___block_1_2, getattr_l__mod___features___10___block_2_0], Original ATen: [aten.convolution, aten.hardswish]
    buf87 = extern_kernels.convolution(buf86, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf87, (4, 80, 14, 14), (15680, 1, 1120, 80))
    del arg99_1
    del buf86
    buf88 = buf80; del buf80  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_36(c_void_p(buf88.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    del arg261_1
    del arg262_1
    del buf87
    # Source Nodes: [getattr_l__mod___features___11___block_0_0, result_15, result_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf89 = extern_kernels.convolution(buf88, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf89, (4, 480, 14, 14), (94080, 1, 6720, 480))
    del arg102_1
    del buf88
    buf90 = buf89; del buf89  # reuse
    buf91 = buf90; del buf90  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_37(c_void_p(buf91.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg104_1.data_ptr()))
    del arg103_1
    del arg104_1
    del arg264_1
    del arg265_1
    # Source Nodes: [getattr_l__mod___features___11___block_0_2, getattr_l__mod___features___11___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
    buf92 = extern_kernels.convolution(buf91, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf92, (4, 480, 14, 14), (94080, 1, 6720, 480))
    del arg105_1
    del buf91
    buf93 = buf92; del buf92  # reuse
    buf94 = empty_strided((4, 480, 1, 1), (480, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf95 = reinterpret_tensor(buf94, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf94  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_38(c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg267_1
    del arg268_1
    # Source Nodes: [getattr_l__mod___features___11___block_1_2, scale_15, scale_16], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf96 = extern_kernels.convolution(buf95, arg108_1, arg109_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf96, (4, 120, 1, 1), (120, 1, 120, 120))
    del arg108_1
    del arg109_1
    del buf95
    buf97 = buf96; del buf96  # reuse
    cpp_fused_relu_39(c_void_p(buf97.data_ptr()))
    # Source Nodes: [scale_17, scale_18], Original ATen: [aten.convolution, aten.relu]
    buf98 = extern_kernels.convolution(buf97, arg110_1, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf98, (4, 480, 1, 1), (480, 1, 480, 480))
    del arg110_1
    del arg111_1
    del buf97
    buf99 = buf93; del buf93  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_40(c_void_p(buf99.data_ptr()), c_void_p(buf98.data_ptr()))
    del buf98
    # Source Nodes: [getattr_l__mod___features___11___block_1_2, getattr_l__mod___features___11___block_3_0, mul_3, scale_19], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf100 = extern_kernels.convolution(buf99, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf100, (4, 112, 14, 14), (21952, 1, 1568, 112))
    del arg112_1
    del buf99
    buf101 = buf100; del buf100  # reuse
    cpp_fused__native_batch_norm_legit_no_training_41(c_void_p(buf101.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg114_1.data_ptr()))
    del arg113_1
    del arg114_1
    del arg270_1
    del arg271_1
    # Source Nodes: [getattr_l__mod___features___12___block_0_0], Original ATen: [aten.convolution]
    buf102 = extern_kernels.convolution(buf101, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (4, 672, 14, 14), (131712, 1, 9408, 672))
    del arg115_1
    buf103 = buf102; del buf102  # reuse
    buf104 = buf103; del buf103  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_42(c_void_p(buf104.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()))
    del arg116_1
    del arg117_1
    del arg273_1
    del arg274_1
    # Source Nodes: [getattr_l__mod___features___12___block_0_2, getattr_l__mod___features___12___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
    buf105 = extern_kernels.convolution(buf104, arg118_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf105, (4, 672, 14, 14), (131712, 1, 9408, 672))
    del arg118_1
    del buf104
    buf106 = buf105; del buf105  # reuse
    buf107 = empty_strided((4, 672, 1, 1), (672, 1, 2688, 2688), device='cpu', dtype=torch.float32)
    buf108 = reinterpret_tensor(buf107, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf107  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_43(c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(arg120_1.data_ptr()))
    del arg119_1
    del arg120_1
    del arg276_1
    del arg277_1
    # Source Nodes: [getattr_l__mod___features___12___block_1_2, scale_20, scale_21], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf109 = extern_kernels.convolution(buf108, arg121_1, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf109, (4, 168, 1, 1), (168, 1, 168, 168))
    del arg121_1
    del arg122_1
    del buf108
    buf110 = buf109; del buf109  # reuse
    cpp_fused_relu_44(c_void_p(buf110.data_ptr()))
    # Source Nodes: [scale_22, scale_23], Original ATen: [aten.convolution, aten.relu]
    buf111 = extern_kernels.convolution(buf110, arg123_1, arg124_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf111, (4, 672, 1, 1), (672, 1, 672, 672))
    del arg123_1
    del arg124_1
    del buf110
    buf112 = buf106; del buf106  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_45(c_void_p(buf112.data_ptr()), c_void_p(buf111.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___12___block_1_2, getattr_l__mod___features___12___block_3_0, mul_4, scale_24], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf113 = extern_kernels.convolution(buf112, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf113, (4, 112, 14, 14), (21952, 1, 1568, 112))
    del arg125_1
    del buf112
    buf114 = buf101; del buf101  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_46(c_void_p(buf114.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()))
    del arg126_1
    del arg127_1
    del arg279_1
    del arg280_1
    del buf113
    # Source Nodes: [getattr_l__mod___features___13___block_0_0, result_18, result_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf115 = extern_kernels.convolution(buf114, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf115, (4, 672, 14, 14), (131712, 1, 9408, 672))
    del arg128_1
    del buf114
    buf116 = buf115; del buf115  # reuse
    buf117 = buf116; del buf116  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_47(c_void_p(buf117.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(arg130_1.data_ptr()))
    del arg129_1
    del arg130_1
    del arg282_1
    del arg283_1
    # Source Nodes: [getattr_l__mod___features___13___block_0_2, getattr_l__mod___features___13___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
    buf118 = extern_kernels.convolution(buf117, arg131_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf118, (4, 672, 7, 7), (32928, 1, 4704, 672))
    del arg131_1
    del buf117
    buf119 = buf118; del buf118  # reuse
    buf120 = reinterpret_tensor(buf111, (4, 672, 1, 1), (672, 1, 2688, 2688), 0); del buf111  # reuse
    buf121 = reinterpret_tensor(buf120, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf120  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_48(c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()))
    del arg132_1
    del arg133_1
    del arg285_1
    del arg286_1
    # Source Nodes: [getattr_l__mod___features___13___block_1_2, scale_25, scale_26], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf122 = extern_kernels.convolution(buf121, arg134_1, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf122, (4, 168, 1, 1), (168, 1, 168, 168))
    del arg134_1
    del arg135_1
    del buf121
    buf123 = buf122; del buf122  # reuse
    cpp_fused_relu_49(c_void_p(buf123.data_ptr()))
    # Source Nodes: [scale_27, scale_28], Original ATen: [aten.convolution, aten.relu]
    buf124 = extern_kernels.convolution(buf123, arg136_1, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf124, (4, 672, 1, 1), (672, 1, 672, 672))
    del arg136_1
    del arg137_1
    del buf123
    buf125 = buf119; del buf119  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_50(c_void_p(buf125.data_ptr()), c_void_p(buf124.data_ptr()))
    del buf124
    # Source Nodes: [getattr_l__mod___features___13___block_1_2, getattr_l__mod___features___13___block_3_0, mul_5, scale_29], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf126 = extern_kernels.convolution(buf125, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf126, (4, 160, 7, 7), (7840, 1, 1120, 160))
    del arg138_1
    del buf125
    buf127 = buf126; del buf126  # reuse
    cpp_fused__native_batch_norm_legit_no_training_51(c_void_p(buf127.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()))
    del arg139_1
    del arg140_1
    del arg288_1
    del arg289_1
    # Source Nodes: [getattr_l__mod___features___14___block_0_0], Original ATen: [aten.convolution]
    buf128 = extern_kernels.convolution(buf127, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf128, (4, 960, 7, 7), (47040, 1, 6720, 960))
    del arg141_1
    buf129 = buf128; del buf128  # reuse
    buf130 = buf129; del buf129  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_52(c_void_p(buf130.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()))
    del arg142_1
    del arg143_1
    del arg291_1
    del arg292_1
    # Source Nodes: [getattr_l__mod___features___14___block_0_2, getattr_l__mod___features___14___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
    buf131 = extern_kernels.convolution(buf130, arg144_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
    assert_size_stride(buf131, (4, 960, 7, 7), (47040, 1, 6720, 960))
    del arg144_1
    del buf130
    buf132 = buf131; del buf131  # reuse
    buf133 = empty_strided((4, 960, 1, 1), (960, 1, 3840, 3840), device='cpu', dtype=torch.float32)
    buf134 = reinterpret_tensor(buf133, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf133  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_53(c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()))
    del arg145_1
    del arg146_1
    del arg294_1
    del arg295_1
    # Source Nodes: [getattr_l__mod___features___14___block_1_2, scale_30, scale_31], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf135 = extern_kernels.convolution(buf134, arg147_1, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf135, (4, 240, 1, 1), (240, 1, 240, 240))
    del arg147_1
    del arg148_1
    del buf134
    buf136 = buf135; del buf135  # reuse
    cpp_fused_relu_54(c_void_p(buf136.data_ptr()))
    # Source Nodes: [scale_32, scale_33], Original ATen: [aten.convolution, aten.relu]
    buf137 = extern_kernels.convolution(buf136, arg149_1, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf137, (4, 960, 1, 1), (960, 1, 960, 960))
    del arg149_1
    del arg150_1
    del buf136
    buf138 = buf132; del buf132  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_55(c_void_p(buf138.data_ptr()), c_void_p(buf137.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___14___block_1_2, getattr_l__mod___features___14___block_3_0, mul_6, scale_34], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf139 = extern_kernels.convolution(buf138, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf139, (4, 160, 7, 7), (7840, 1, 1120, 160))
    del arg151_1
    del buf138
    buf140 = buf127; del buf127  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_56(c_void_p(buf140.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(arg153_1.data_ptr()))
    del arg152_1
    del arg153_1
    del arg297_1
    del arg298_1
    del buf139
    # Source Nodes: [getattr_l__mod___features___15___block_0_0], Original ATen: [aten.convolution]
    buf141 = extern_kernels.convolution(buf140, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf141, (4, 960, 7, 7), (47040, 1, 6720, 960))
    del arg154_1
    buf142 = buf141; del buf141  # reuse
    buf143 = buf142; del buf142  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_57(c_void_p(buf143.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(arg156_1.data_ptr()))
    del arg155_1
    del arg156_1
    del arg300_1
    del arg301_1
    # Source Nodes: [getattr_l__mod___features___15___block_0_2, getattr_l__mod___features___15___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
    buf144 = extern_kernels.convolution(buf143, arg157_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
    assert_size_stride(buf144, (4, 960, 7, 7), (47040, 1, 6720, 960))
    del arg157_1
    del buf143
    buf145 = buf144; del buf144  # reuse
    buf146 = reinterpret_tensor(buf137, (4, 960, 1, 1), (960, 1, 3840, 3840), 0); del buf137  # reuse
    buf147 = reinterpret_tensor(buf146, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf146  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_58(c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg159_1.data_ptr()))
    del arg158_1
    del arg159_1
    del arg303_1
    del arg304_1
    # Source Nodes: [getattr_l__mod___features___15___block_1_2, scale_35, scale_36], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf148 = extern_kernels.convolution(buf147, arg160_1, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf148, (4, 240, 1, 1), (240, 1, 240, 240))
    del arg160_1
    del arg161_1
    del buf147
    buf149 = buf148; del buf148  # reuse
    cpp_fused_relu_59(c_void_p(buf149.data_ptr()))
    # Source Nodes: [scale_37, scale_38], Original ATen: [aten.convolution, aten.relu]
    buf150 = extern_kernels.convolution(buf149, arg162_1, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf150, (4, 960, 1, 1), (960, 1, 960, 960))
    del arg162_1
    del arg163_1
    del buf149
    buf151 = buf145; del buf145  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_60(c_void_p(buf151.data_ptr()), c_void_p(buf150.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___15___block_1_2, getattr_l__mod___features___15___block_3_0, mul_7, scale_39], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf152 = extern_kernels.convolution(buf151, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf152, (4, 160, 7, 7), (7840, 1, 1120, 160))
    del arg164_1
    del buf151
    buf153 = buf140; del buf140  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_61(c_void_p(buf153.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg166_1.data_ptr()))
    del arg165_1
    del arg166_1
    del arg306_1
    del arg307_1
    del buf152
    # Source Nodes: [l__mod___features_16_0, result_23, result_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf154 = extern_kernels.convolution(buf153, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf154, (4, 960, 7, 7), (47040, 1, 6720, 960))
    del arg167_1
    del buf153
    buf155 = buf154; del buf154  # reuse
    buf156 = reinterpret_tensor(buf150, (4, 960, 1, 1), (960, 1, 3840, 3840), 0); del buf150  # reuse
    buf157 = reinterpret_tensor(buf156, (4, 960, 1, 1), (960, 1, 1, 1), 0); del buf156  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_62(c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()))
    del arg168_1
    del arg169_1
    del arg309_1
    del arg310_1
    del buf155
    buf158 = empty((4, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___classifier_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg171_1, reinterpret_tensor(buf157, (4, 960), (960, 1), 0), reinterpret_tensor(arg170_1, (960, 1280), (1, 960), 0), alpha=1, beta=1, out=buf158)
    del arg170_1
    del arg171_1
    del buf157
    buf159 = buf158; del buf158  # reuse
    cpp_fused_hardswish_63(c_void_p(buf159.data_ptr()))
    buf160 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___classifier_1, x_3], Original ATen: [aten.addmm, aten.hardswish]
    extern_kernels.addmm(arg173_1, buf159, reinterpret_tensor(arg172_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf160)
    del arg172_1
    del arg173_1
    return (buf160, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((200, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((200, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((80, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((672, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((168, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((168, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((160, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((1280, 960), (960, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg177_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg180_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg183_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg186_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg189_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg192_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg195_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg198_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg201_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg204_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg207_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg210_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg213_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg216_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg219_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg222_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg225_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg228_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg231_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg234_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg237_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg240_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg243_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg246_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg249_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg252_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg255_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg258_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg261_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg264_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg267_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg270_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg273_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg276_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg279_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg282_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg285_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg288_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg291_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg294_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg297_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg300_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg303_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg306_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg309_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg312_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenet_v3_large', benchmark_compiled_module)
