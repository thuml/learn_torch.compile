
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


cpp_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_add_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_31 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_34 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_37 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_58 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (352L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (352L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1984L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1984L*x2) + (97216L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                            tmp_acc0_vec = tmp_acc0_vec + tmp17;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1984L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(15872L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, ), (1, ))
    assert_size_stride(arg1_1, (16, ), (1, ))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, ), (1, ))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (96, ), (1, ))
    assert_size_stride(arg9_1, (96, ), (1, ))
    assert_size_stride(arg10_1, (96, ), (1, ))
    assert_size_stride(arg11_1, (96, ), (1, ))
    assert_size_stride(arg12_1, (24, ), (1, ))
    assert_size_stride(arg13_1, (24, ), (1, ))
    assert_size_stride(arg14_1, (24, ), (1, ))
    assert_size_stride(arg15_1, (24, ), (1, ))
    assert_size_stride(arg16_1, (24, ), (1, ))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (24, ), (1, ))
    assert_size_stride(arg19_1, (24, ), (1, ))
    assert_size_stride(arg20_1, (24, ), (1, ))
    assert_size_stride(arg21_1, (24, ), (1, ))
    assert_size_stride(arg22_1, (24, ), (1, ))
    assert_size_stride(arg23_1, (24, ), (1, ))
    assert_size_stride(arg24_1, (24, ), (1, ))
    assert_size_stride(arg25_1, (24, ), (1, ))
    assert_size_stride(arg26_1, (144, ), (1, ))
    assert_size_stride(arg27_1, (144, ), (1, ))
    assert_size_stride(arg28_1, (144, ), (1, ))
    assert_size_stride(arg29_1, (144, ), (1, ))
    assert_size_stride(arg30_1, (32, ), (1, ))
    assert_size_stride(arg31_1, (32, ), (1, ))
    assert_size_stride(arg32_1, (96, ), (1, ))
    assert_size_stride(arg33_1, (96, ), (1, ))
    assert_size_stride(arg34_1, (96, ), (1, ))
    assert_size_stride(arg35_1, (96, ), (1, ))
    assert_size_stride(arg36_1, (32, ), (1, ))
    assert_size_stride(arg37_1, (32, ), (1, ))
    assert_size_stride(arg38_1, (192, ), (1, ))
    assert_size_stride(arg39_1, (192, ), (1, ))
    assert_size_stride(arg40_1, (192, ), (1, ))
    assert_size_stride(arg41_1, (192, ), (1, ))
    assert_size_stride(arg42_1, (32, ), (1, ))
    assert_size_stride(arg43_1, (32, ), (1, ))
    assert_size_stride(arg44_1, (192, ), (1, ))
    assert_size_stride(arg45_1, (192, ), (1, ))
    assert_size_stride(arg46_1, (192, ), (1, ))
    assert_size_stride(arg47_1, (192, ), (1, ))
    assert_size_stride(arg48_1, (32, ), (1, ))
    assert_size_stride(arg49_1, (32, ), (1, ))
    assert_size_stride(arg50_1, (192, ), (1, ))
    assert_size_stride(arg51_1, (192, ), (1, ))
    assert_size_stride(arg52_1, (192, ), (1, ))
    assert_size_stride(arg53_1, (192, ), (1, ))
    assert_size_stride(arg54_1, (64, ), (1, ))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (192, ), (1, ))
    assert_size_stride(arg57_1, (192, ), (1, ))
    assert_size_stride(arg58_1, (192, ), (1, ))
    assert_size_stride(arg59_1, (192, ), (1, ))
    assert_size_stride(arg60_1, (64, ), (1, ))
    assert_size_stride(arg61_1, (64, ), (1, ))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (384, ), (1, ))
    assert_size_stride(arg64_1, (384, ), (1, ))
    assert_size_stride(arg65_1, (384, ), (1, ))
    assert_size_stride(arg66_1, (64, ), (1, ))
    assert_size_stride(arg67_1, (64, ), (1, ))
    assert_size_stride(arg68_1, (384, ), (1, ))
    assert_size_stride(arg69_1, (384, ), (1, ))
    assert_size_stride(arg70_1, (384, ), (1, ))
    assert_size_stride(arg71_1, (384, ), (1, ))
    assert_size_stride(arg72_1, (64, ), (1, ))
    assert_size_stride(arg73_1, (64, ), (1, ))
    assert_size_stride(arg74_1, (384, ), (1, ))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (384, ), (1, ))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (112, ), (1, ))
    assert_size_stride(arg79_1, (112, ), (1, ))
    assert_size_stride(arg80_1, (672, ), (1, ))
    assert_size_stride(arg81_1, (672, ), (1, ))
    assert_size_stride(arg82_1, (672, ), (1, ))
    assert_size_stride(arg83_1, (672, ), (1, ))
    assert_size_stride(arg84_1, (112, ), (1, ))
    assert_size_stride(arg85_1, (112, ), (1, ))
    assert_size_stride(arg86_1, (672, ), (1, ))
    assert_size_stride(arg87_1, (672, ), (1, ))
    assert_size_stride(arg88_1, (672, ), (1, ))
    assert_size_stride(arg89_1, (672, ), (1, ))
    assert_size_stride(arg90_1, (112, ), (1, ))
    assert_size_stride(arg91_1, (112, ), (1, ))
    assert_size_stride(arg92_1, (336, ), (1, ))
    assert_size_stride(arg93_1, (336, ), (1, ))
    assert_size_stride(arg94_1, (336, ), (1, ))
    assert_size_stride(arg95_1, (336, ), (1, ))
    assert_size_stride(arg96_1, (112, ), (1, ))
    assert_size_stride(arg97_1, (112, ), (1, ))
    assert_size_stride(arg98_1, (672, ), (1, ))
    assert_size_stride(arg99_1, (672, ), (1, ))
    assert_size_stride(arg100_1, (672, ), (1, ))
    assert_size_stride(arg101_1, (672, ), (1, ))
    assert_size_stride(arg102_1, (184, ), (1, ))
    assert_size_stride(arg103_1, (184, ), (1, ))
    assert_size_stride(arg104_1, (1104, ), (1, ))
    assert_size_stride(arg105_1, (1104, ), (1, ))
    assert_size_stride(arg106_1, (1104, ), (1, ))
    assert_size_stride(arg107_1, (1104, ), (1, ))
    assert_size_stride(arg108_1, (184, ), (1, ))
    assert_size_stride(arg109_1, (184, ), (1, ))
    assert_size_stride(arg110_1, (1104, ), (1, ))
    assert_size_stride(arg111_1, (1104, ), (1, ))
    assert_size_stride(arg112_1, (1104, ), (1, ))
    assert_size_stride(arg113_1, (1104, ), (1, ))
    assert_size_stride(arg114_1, (184, ), (1, ))
    assert_size_stride(arg115_1, (184, ), (1, ))
    assert_size_stride(arg116_1, (1104, ), (1, ))
    assert_size_stride(arg117_1, (1104, ), (1, ))
    assert_size_stride(arg118_1, (1104, ), (1, ))
    assert_size_stride(arg119_1, (1104, ), (1, ))
    assert_size_stride(arg120_1, (184, ), (1, ))
    assert_size_stride(arg121_1, (184, ), (1, ))
    assert_size_stride(arg122_1, (1104, ), (1, ))
    assert_size_stride(arg123_1, (1104, ), (1, ))
    assert_size_stride(arg124_1, (1104, ), (1, ))
    assert_size_stride(arg125_1, (1104, ), (1, ))
    assert_size_stride(arg126_1, (352, ), (1, ))
    assert_size_stride(arg127_1, (352, ), (1, ))
    assert_size_stride(arg128_1, (1984, ), (1, ))
    assert_size_stride(arg129_1, (1984, ), (1, ))
    assert_size_stride(arg130_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg131_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg132_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg133_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg134_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg135_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg136_1, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg137_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg138_1, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg139_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg140_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg141_1, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg142_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg143_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg144_1, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg145_1, (32, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg146_1, (96, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg147_1, (96, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg148_1, (32, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg149_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg150_1, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg151_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg152_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg153_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg154_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg155_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg156_1, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg157_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg158_1, (192, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg159_1, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg160_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg161_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg162_1, (384, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg163_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg164_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg165_1, (384, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg166_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg167_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg168_1, (384, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg169_1, (112, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg170_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg171_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg172_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg173_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg174_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg175_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg176_1, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg177_1, (336, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg178_1, (112, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg179_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg180_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg181_1, (184, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg182_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg183_1, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg184_1, (184, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg185_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg186_1, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg187_1, (184, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg188_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg189_1, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg190_1, (184, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg191_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg192_1, (1104, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg193_1, (352, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg194_1, (1984, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(arg195_1, (1000, 1984), (1984, 1))
    assert_size_stride(arg196_1, (1000, ), (1, ))
    assert_size_stride(arg197_1, (16, ), (1, ))
    assert_size_stride(arg198_1, (16, ), (1, ))
    assert_size_stride(arg199_1, (16, ), (1, ))
    assert_size_stride(arg200_1, (16, ), (1, ))
    assert_size_stride(arg201_1, (16, ), (1, ))
    assert_size_stride(arg202_1, (16, ), (1, ))
    assert_size_stride(arg203_1, (16, ), (1, ))
    assert_size_stride(arg204_1, (16, ), (1, ))
    assert_size_stride(arg205_1, (96, ), (1, ))
    assert_size_stride(arg206_1, (96, ), (1, ))
    assert_size_stride(arg207_1, (96, ), (1, ))
    assert_size_stride(arg208_1, (96, ), (1, ))
    assert_size_stride(arg209_1, (24, ), (1, ))
    assert_size_stride(arg210_1, (24, ), (1, ))
    assert_size_stride(arg211_1, (24, ), (1, ))
    assert_size_stride(arg212_1, (24, ), (1, ))
    assert_size_stride(arg213_1, (24, ), (1, ))
    assert_size_stride(arg214_1, (24, ), (1, ))
    assert_size_stride(arg215_1, (24, ), (1, ))
    assert_size_stride(arg216_1, (24, ), (1, ))
    assert_size_stride(arg217_1, (24, ), (1, ))
    assert_size_stride(arg218_1, (24, ), (1, ))
    assert_size_stride(arg219_1, (24, ), (1, ))
    assert_size_stride(arg220_1, (24, ), (1, ))
    assert_size_stride(arg221_1, (24, ), (1, ))
    assert_size_stride(arg222_1, (24, ), (1, ))
    assert_size_stride(arg223_1, (144, ), (1, ))
    assert_size_stride(arg224_1, (144, ), (1, ))
    assert_size_stride(arg225_1, (144, ), (1, ))
    assert_size_stride(arg226_1, (144, ), (1, ))
    assert_size_stride(arg227_1, (32, ), (1, ))
    assert_size_stride(arg228_1, (32, ), (1, ))
    assert_size_stride(arg229_1, (96, ), (1, ))
    assert_size_stride(arg230_1, (96, ), (1, ))
    assert_size_stride(arg231_1, (96, ), (1, ))
    assert_size_stride(arg232_1, (96, ), (1, ))
    assert_size_stride(arg233_1, (32, ), (1, ))
    assert_size_stride(arg234_1, (32, ), (1, ))
    assert_size_stride(arg235_1, (192, ), (1, ))
    assert_size_stride(arg236_1, (192, ), (1, ))
    assert_size_stride(arg237_1, (192, ), (1, ))
    assert_size_stride(arg238_1, (192, ), (1, ))
    assert_size_stride(arg239_1, (32, ), (1, ))
    assert_size_stride(arg240_1, (32, ), (1, ))
    assert_size_stride(arg241_1, (192, ), (1, ))
    assert_size_stride(arg242_1, (192, ), (1, ))
    assert_size_stride(arg243_1, (192, ), (1, ))
    assert_size_stride(arg244_1, (192, ), (1, ))
    assert_size_stride(arg245_1, (32, ), (1, ))
    assert_size_stride(arg246_1, (32, ), (1, ))
    assert_size_stride(arg247_1, (192, ), (1, ))
    assert_size_stride(arg248_1, (192, ), (1, ))
    assert_size_stride(arg249_1, (192, ), (1, ))
    assert_size_stride(arg250_1, (192, ), (1, ))
    assert_size_stride(arg251_1, (64, ), (1, ))
    assert_size_stride(arg252_1, (64, ), (1, ))
    assert_size_stride(arg253_1, (192, ), (1, ))
    assert_size_stride(arg254_1, (192, ), (1, ))
    assert_size_stride(arg255_1, (192, ), (1, ))
    assert_size_stride(arg256_1, (192, ), (1, ))
    assert_size_stride(arg257_1, (64, ), (1, ))
    assert_size_stride(arg258_1, (64, ), (1, ))
    assert_size_stride(arg259_1, (384, ), (1, ))
    assert_size_stride(arg260_1, (384, ), (1, ))
    assert_size_stride(arg261_1, (384, ), (1, ))
    assert_size_stride(arg262_1, (384, ), (1, ))
    assert_size_stride(arg263_1, (64, ), (1, ))
    assert_size_stride(arg264_1, (64, ), (1, ))
    assert_size_stride(arg265_1, (384, ), (1, ))
    assert_size_stride(arg266_1, (384, ), (1, ))
    assert_size_stride(arg267_1, (384, ), (1, ))
    assert_size_stride(arg268_1, (384, ), (1, ))
    assert_size_stride(arg269_1, (64, ), (1, ))
    assert_size_stride(arg270_1, (64, ), (1, ))
    assert_size_stride(arg271_1, (384, ), (1, ))
    assert_size_stride(arg272_1, (384, ), (1, ))
    assert_size_stride(arg273_1, (384, ), (1, ))
    assert_size_stride(arg274_1, (384, ), (1, ))
    assert_size_stride(arg275_1, (112, ), (1, ))
    assert_size_stride(arg276_1, (112, ), (1, ))
    assert_size_stride(arg277_1, (672, ), (1, ))
    assert_size_stride(arg278_1, (672, ), (1, ))
    assert_size_stride(arg279_1, (672, ), (1, ))
    assert_size_stride(arg280_1, (672, ), (1, ))
    assert_size_stride(arg281_1, (112, ), (1, ))
    assert_size_stride(arg282_1, (112, ), (1, ))
    assert_size_stride(arg283_1, (672, ), (1, ))
    assert_size_stride(arg284_1, (672, ), (1, ))
    assert_size_stride(arg285_1, (672, ), (1, ))
    assert_size_stride(arg286_1, (672, ), (1, ))
    assert_size_stride(arg287_1, (112, ), (1, ))
    assert_size_stride(arg288_1, (112, ), (1, ))
    assert_size_stride(arg289_1, (336, ), (1, ))
    assert_size_stride(arg290_1, (336, ), (1, ))
    assert_size_stride(arg291_1, (336, ), (1, ))
    assert_size_stride(arg292_1, (336, ), (1, ))
    assert_size_stride(arg293_1, (112, ), (1, ))
    assert_size_stride(arg294_1, (112, ), (1, ))
    assert_size_stride(arg295_1, (672, ), (1, ))
    assert_size_stride(arg296_1, (672, ), (1, ))
    assert_size_stride(arg297_1, (672, ), (1, ))
    assert_size_stride(arg298_1, (672, ), (1, ))
    assert_size_stride(arg299_1, (184, ), (1, ))
    assert_size_stride(arg300_1, (184, ), (1, ))
    assert_size_stride(arg301_1, (1104, ), (1, ))
    assert_size_stride(arg302_1, (1104, ), (1, ))
    assert_size_stride(arg303_1, (1104, ), (1, ))
    assert_size_stride(arg304_1, (1104, ), (1, ))
    assert_size_stride(arg305_1, (184, ), (1, ))
    assert_size_stride(arg306_1, (184, ), (1, ))
    assert_size_stride(arg307_1, (1104, ), (1, ))
    assert_size_stride(arg308_1, (1104, ), (1, ))
    assert_size_stride(arg309_1, (1104, ), (1, ))
    assert_size_stride(arg310_1, (1104, ), (1, ))
    assert_size_stride(arg311_1, (184, ), (1, ))
    assert_size_stride(arg312_1, (184, ), (1, ))
    assert_size_stride(arg313_1, (1104, ), (1, ))
    assert_size_stride(arg314_1, (1104, ), (1, ))
    assert_size_stride(arg315_1, (1104, ), (1, ))
    assert_size_stride(arg316_1, (1104, ), (1, ))
    assert_size_stride(arg317_1, (184, ), (1, ))
    assert_size_stride(arg318_1, (184, ), (1, ))
    assert_size_stride(arg319_1, (1104, ), (1, ))
    assert_size_stride(arg320_1, (1104, ), (1, ))
    assert_size_stride(arg321_1, (1104, ), (1, ))
    assert_size_stride(arg322_1, (1104, ), (1, ))
    assert_size_stride(arg323_1, (352, ), (1, ))
    assert_size_stride(arg324_1, (352, ), (1, ))
    assert_size_stride(arg325_1, (1984, ), (1, ))
    assert_size_stride(arg326_1, (1984, ), (1, ))
    assert_size_stride(arg327_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg327_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg130_1
    del arg327_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg0_1
    del arg197_1
    del arg198_1
    del arg1_1
    # Source Nodes: [x_5], Original ATen: [aten.convolution]
    buf4 = extern_kernels.convolution(buf3, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf4, (8, 16, 112, 112), (200704, 1, 1792, 16))
    del arg131_1
    buf5 = buf4; del buf4  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_2(c_void_p(buf5.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()))
    del arg199_1
    del arg200_1
    del arg2_1
    del arg3_1
    # Source Nodes: [x_10, x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf6 = extern_kernels.convolution(buf5, arg132_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
    assert_size_stride(buf6, (8, 16, 112, 112), (200704, 1, 1792, 16))
    del arg132_1
    del buf5
    buf7 = buf6; del buf6  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_3(c_void_p(buf7.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg201_1
    del arg202_1
    del arg4_1
    del arg5_1
    # Source Nodes: [x_11, x_14, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf8 = extern_kernels.convolution(buf7, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (8, 16, 112, 112), (200704, 1, 1792, 16))
    del arg133_1
    del buf7
    buf9 = buf3; del buf3  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_4(c_void_p(buf9.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()))
    del arg203_1
    del arg204_1
    del arg6_1
    del arg7_1
    del buf8
    # Source Nodes: [shortcut_1, x_17, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf10 = extern_kernels.convolution(buf9, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf10, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    del arg134_1
    del buf9
    buf11 = buf10; del buf10  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_5(c_void_p(buf11.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg205_1
    del arg206_1
    del arg8_1
    del arg9_1
    # Source Nodes: [x_23, x_26, x_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf12 = extern_kernels.convolution(buf11, arg135_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
    assert_size_stride(buf12, (8, 96, 56, 56), (301056, 1, 5376, 96))
    del arg135_1
    del buf11
    buf13 = buf12; del buf12  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_6(c_void_p(buf13.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()))
    del arg10_1
    del arg11_1
    del arg207_1
    del arg208_1
    # Source Nodes: [x_28, x_31, x_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf14 = extern_kernels.convolution(buf13, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf14, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg136_1
    del buf13
    buf15 = buf14; del buf14  # reuse
    cpp_fused__native_batch_norm_legit_no_training_7(c_void_p(buf15.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()))
    del arg12_1
    del arg13_1
    del arg209_1
    del arg210_1
    # Source Nodes: [x_38], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf15, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf16, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg137_1
    buf17 = buf16; del buf16  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_8(c_void_p(buf17.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()))
    del arg14_1
    del arg15_1
    del arg211_1
    del arg212_1
    # Source Nodes: [x_39, x_42, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf18 = extern_kernels.convolution(buf17, arg138_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
    assert_size_stride(buf18, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg138_1
    del buf17
    buf19 = buf18; del buf18  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_9(c_void_p(buf19.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg16_1
    del arg17_1
    del arg213_1
    del arg214_1
    # Source Nodes: [x_44, x_47, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf20 = extern_kernels.convolution(buf19, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf20, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg139_1
    del buf19
    buf21 = buf15; del buf15  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_10(c_void_p(buf21.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()))
    del arg18_1
    del arg19_1
    del arg215_1
    del arg216_1
    del buf20
    # Source Nodes: [x_55], Original ATen: [aten.convolution]
    buf22 = extern_kernels.convolution(buf21, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf22, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg140_1
    buf23 = buf22; del buf22  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_11(c_void_p(buf23.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()))
    del arg20_1
    del arg217_1
    del arg218_1
    del arg21_1
    # Source Nodes: [x_56, x_59, x_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf24 = extern_kernels.convolution(buf23, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
    assert_size_stride(buf24, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg141_1
    del buf23
    buf25 = buf24; del buf24  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_12(c_void_p(buf25.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg219_1
    del arg220_1
    del arg22_1
    del arg23_1
    # Source Nodes: [x_61, x_64, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf26 = extern_kernels.convolution(buf25, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf26, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg142_1
    del buf25
    buf27 = buf21; del buf21  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_13(c_void_p(buf27.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()))
    del arg221_1
    del arg222_1
    del arg24_1
    del arg25_1
    del buf26
    # Source Nodes: [shortcut_4, x_67, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf28 = extern_kernels.convolution(buf27, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf28, (8, 144, 56, 56), (451584, 1, 8064, 144))
    del arg143_1
    del buf27
    buf29 = buf28; del buf28  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_14(c_void_p(buf29.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()))
    del arg223_1
    del arg224_1
    del arg26_1
    del arg27_1
    # Source Nodes: [x_73, x_76, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf30 = extern_kernels.convolution(buf29, arg144_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
    assert_size_stride(buf30, (8, 144, 28, 28), (112896, 1, 4032, 144))
    del arg144_1
    del buf29
    buf31 = buf30; del buf30  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_15(c_void_p(buf31.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg225_1
    del arg226_1
    del arg28_1
    del arg29_1
    # Source Nodes: [x_78, x_81, x_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf32 = extern_kernels.convolution(buf31, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf32, (8, 32, 28, 28), (25088, 1, 896, 32))
    del arg145_1
    del buf31
    buf33 = buf32; del buf32  # reuse
    cpp_fused__native_batch_norm_legit_no_training_16(c_void_p(buf33.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()))
    del arg227_1
    del arg228_1
    del arg30_1
    del arg31_1
    # Source Nodes: [x_88], Original ATen: [aten.convolution]
    buf34 = extern_kernels.convolution(buf33, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf34, (8, 96, 28, 28), (75264, 1, 2688, 96))
    del arg146_1
    buf35 = buf34; del buf34  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_17(c_void_p(buf35.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()))
    del arg229_1
    del arg230_1
    del arg32_1
    del arg33_1
    # Source Nodes: [x_89, x_92, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf36 = extern_kernels.convolution(buf35, arg147_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
    assert_size_stride(buf36, (8, 96, 28, 28), (75264, 1, 2688, 96))
    del arg147_1
    del buf35
    buf37 = buf36; del buf36  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_18(c_void_p(buf37.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg231_1
    del arg232_1
    del arg34_1
    del arg35_1
    # Source Nodes: [x_94, x_97, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf38 = extern_kernels.convolution(buf37, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf38, (8, 32, 28, 28), (25088, 1, 896, 32))
    del arg148_1
    del buf37
    buf39 = buf33; del buf33  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_19(c_void_p(buf39.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()))
    del arg233_1
    del arg234_1
    del arg36_1
    del arg37_1
    del buf38
    # Source Nodes: [x_105], Original ATen: [aten.convolution]
    buf40 = extern_kernels.convolution(buf39, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf40, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del arg149_1
    buf41 = buf40; del buf40  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_20(c_void_p(buf41.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()))
    del arg235_1
    del arg236_1
    del arg38_1
    del arg39_1
    # Source Nodes: [x_106, x_109, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf42 = extern_kernels.convolution(buf41, arg150_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
    assert_size_stride(buf42, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del arg150_1
    del buf41
    buf43 = buf42; del buf42  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_21(c_void_p(buf43.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg237_1
    del arg238_1
    del arg40_1
    del arg41_1
    # Source Nodes: [x_111, x_114, x_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf44 = extern_kernels.convolution(buf43, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf44, (8, 32, 28, 28), (25088, 1, 896, 32))
    del arg151_1
    del buf43
    buf45 = buf39; del buf39  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_22(c_void_p(buf45.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()))
    del arg239_1
    del arg240_1
    del arg42_1
    del arg43_1
    del buf44
    # Source Nodes: [x_122], Original ATen: [aten.convolution]
    buf46 = extern_kernels.convolution(buf45, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf46, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del arg152_1
    buf47 = buf46; del buf46  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_23(c_void_p(buf47.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()))
    del arg241_1
    del arg242_1
    del arg44_1
    del arg45_1
    # Source Nodes: [x_123, x_126, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf48 = extern_kernels.convolution(buf47, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
    assert_size_stride(buf48, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del arg153_1
    del buf47
    buf49 = buf48; del buf48  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_24(c_void_p(buf49.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg243_1
    del arg244_1
    del arg46_1
    del arg47_1
    # Source Nodes: [x_128, x_131, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf50 = extern_kernels.convolution(buf49, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf50, (8, 32, 28, 28), (25088, 1, 896, 32))
    del arg154_1
    del buf49
    buf51 = buf45; del buf45  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_25(c_void_p(buf51.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()))
    del arg245_1
    del arg246_1
    del arg48_1
    del arg49_1
    del buf50
    # Source Nodes: [shortcut_8, x_134, x_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf52 = extern_kernels.convolution(buf51, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf52, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del arg155_1
    del buf51
    buf53 = buf52; del buf52  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_26(c_void_p(buf53.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()))
    del arg247_1
    del arg248_1
    del arg50_1
    del arg51_1
    # Source Nodes: [x_140, x_143, x_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf54 = extern_kernels.convolution(buf53, arg156_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
    assert_size_stride(buf54, (8, 192, 14, 14), (37632, 1, 2688, 192))
    del arg156_1
    del buf53
    buf55 = buf54; del buf54  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_27(c_void_p(buf55.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg249_1
    del arg250_1
    del arg52_1
    del arg53_1
    # Source Nodes: [x_145, x_148, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf56 = extern_kernels.convolution(buf55, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf56, (8, 64, 14, 14), (12544, 1, 896, 64))
    del arg157_1
    del buf55
    buf57 = buf56; del buf56  # reuse
    cpp_fused__native_batch_norm_legit_no_training_28(c_void_p(buf57.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()))
    del arg251_1
    del arg252_1
    del arg54_1
    del arg55_1
    # Source Nodes: [x_155], Original ATen: [aten.convolution]
    buf58 = extern_kernels.convolution(buf57, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf58, (8, 192, 14, 14), (37632, 1, 2688, 192))
    del arg158_1
    buf59 = buf58; del buf58  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_29(c_void_p(buf59.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()))
    del arg253_1
    del arg254_1
    del arg56_1
    del arg57_1
    # Source Nodes: [x_156, x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf60 = extern_kernels.convolution(buf59, arg159_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
    assert_size_stride(buf60, (8, 192, 14, 14), (37632, 1, 2688, 192))
    del arg159_1
    del buf59
    buf61 = buf60; del buf60  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_30(c_void_p(buf61.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg255_1
    del arg256_1
    del arg58_1
    del arg59_1
    # Source Nodes: [x_161, x_164, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf62 = extern_kernels.convolution(buf61, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf62, (8, 64, 14, 14), (12544, 1, 896, 64))
    del arg160_1
    del buf61
    buf63 = buf57; del buf57  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_31(c_void_p(buf63.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()))
    del arg257_1
    del arg258_1
    del arg60_1
    del arg61_1
    del buf62
    # Source Nodes: [x_172], Original ATen: [aten.convolution]
    buf64 = extern_kernels.convolution(buf63, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf64, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg161_1
    buf65 = buf64; del buf64  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_32(c_void_p(buf65.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()))
    del arg259_1
    del arg260_1
    del arg62_1
    del arg63_1
    # Source Nodes: [x_173, x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf66 = extern_kernels.convolution(buf65, arg162_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
    assert_size_stride(buf66, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg162_1
    del buf65
    buf67 = buf66; del buf66  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_33(c_void_p(buf67.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg261_1
    del arg262_1
    del arg64_1
    del arg65_1
    # Source Nodes: [x_178, x_181, x_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf68 = extern_kernels.convolution(buf67, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (8, 64, 14, 14), (12544, 1, 896, 64))
    del arg163_1
    del buf67
    buf69 = buf63; del buf63  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_34(c_void_p(buf69.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg263_1
    del arg264_1
    del arg66_1
    del arg67_1
    del buf68
    # Source Nodes: [x_189], Original ATen: [aten.convolution]
    buf70 = extern_kernels.convolution(buf69, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf70, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg164_1
    buf71 = buf70; del buf70  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_35(c_void_p(buf71.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()))
    del arg265_1
    del arg266_1
    del arg68_1
    del arg69_1
    # Source Nodes: [x_190, x_193, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf72 = extern_kernels.convolution(buf71, arg165_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
    assert_size_stride(buf72, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg165_1
    del buf71
    buf73 = buf72; del buf72  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_36(c_void_p(buf73.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg267_1
    del arg268_1
    del arg70_1
    del arg71_1
    # Source Nodes: [x_195, x_198, x_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf74 = extern_kernels.convolution(buf73, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf74, (8, 64, 14, 14), (12544, 1, 896, 64))
    del arg166_1
    del buf73
    buf75 = buf69; del buf69  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_37(c_void_p(buf75.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()))
    del arg269_1
    del arg270_1
    del arg72_1
    del arg73_1
    del buf74
    # Source Nodes: [shortcut_12, x_201, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf76 = extern_kernels.convolution(buf75, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf76, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg167_1
    del buf75
    buf77 = buf76; del buf76  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_38(c_void_p(buf77.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()))
    del arg271_1
    del arg272_1
    del arg74_1
    del arg75_1
    # Source Nodes: [x_207, x_210, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf78 = extern_kernels.convolution(buf77, arg168_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
    assert_size_stride(buf78, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg168_1
    del buf77
    buf79 = buf78; del buf78  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_39(c_void_p(buf79.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg273_1
    del arg274_1
    del arg76_1
    del arg77_1
    # Source Nodes: [x_212, x_215, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf80 = extern_kernels.convolution(buf79, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf80, (8, 112, 14, 14), (21952, 1, 1568, 112))
    del arg169_1
    del buf79
    buf81 = buf80; del buf80  # reuse
    cpp_fused__native_batch_norm_legit_no_training_40(c_void_p(buf81.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()))
    del arg275_1
    del arg276_1
    del arg78_1
    del arg79_1
    # Source Nodes: [x_222], Original ATen: [aten.convolution]
    buf82 = extern_kernels.convolution(buf81, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf82, (8, 672, 14, 14), (131712, 1, 9408, 672))
    del arg170_1
    buf83 = buf82; del buf82  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_41(c_void_p(buf83.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()))
    del arg277_1
    del arg278_1
    del arg80_1
    del arg81_1
    # Source Nodes: [x_223, x_226, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf84 = extern_kernels.convolution(buf83, arg171_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf84, (8, 672, 14, 14), (131712, 1, 9408, 672))
    del arg171_1
    del buf83
    buf85 = buf84; del buf84  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_42(c_void_p(buf85.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg279_1
    del arg280_1
    del arg82_1
    del arg83_1
    # Source Nodes: [x_228, x_231, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf86 = extern_kernels.convolution(buf85, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf86, (8, 112, 14, 14), (21952, 1, 1568, 112))
    del arg172_1
    del buf85
    buf87 = buf81; del buf81  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_43(c_void_p(buf87.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()))
    del arg281_1
    del arg282_1
    del arg84_1
    del arg85_1
    del buf86
    # Source Nodes: [x_239], Original ATen: [aten.convolution]
    buf88 = extern_kernels.convolution(buf87, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf88, (8, 672, 14, 14), (131712, 1, 9408, 672))
    del arg173_1
    buf89 = buf88; del buf88  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_44(c_void_p(buf89.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()))
    del arg283_1
    del arg284_1
    del arg86_1
    del arg87_1
    # Source Nodes: [x_240, x_243, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf90 = extern_kernels.convolution(buf89, arg174_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf90, (8, 672, 14, 14), (131712, 1, 9408, 672))
    del arg174_1
    del buf89
    buf91 = buf90; del buf90  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_45(c_void_p(buf91.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg285_1
    del arg286_1
    del arg88_1
    del arg89_1
    # Source Nodes: [x_245, x_248, x_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf92 = extern_kernels.convolution(buf91, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf92, (8, 112, 14, 14), (21952, 1, 1568, 112))
    del arg175_1
    del buf91
    buf93 = buf87; del buf87  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_46(c_void_p(buf93.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()))
    del arg287_1
    del arg288_1
    del arg90_1
    del arg91_1
    del buf92
    # Source Nodes: [x_256], Original ATen: [aten.convolution]
    buf94 = extern_kernels.convolution(buf93, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf94, (8, 336, 14, 14), (65856, 1, 4704, 336))
    del arg176_1
    buf95 = buf94; del buf94  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_47(c_void_p(buf95.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()))
    del arg289_1
    del arg290_1
    del arg92_1
    del arg93_1
    # Source Nodes: [x_257, x_260, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf96 = extern_kernels.convolution(buf95, arg177_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
    assert_size_stride(buf96, (8, 336, 14, 14), (65856, 1, 4704, 336))
    del arg177_1
    del buf95
    buf97 = buf96; del buf96  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_48(c_void_p(buf97.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg291_1
    del arg292_1
    del arg94_1
    del arg95_1
    # Source Nodes: [x_262, x_265, x_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf98 = extern_kernels.convolution(buf97, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf98, (8, 112, 14, 14), (21952, 1, 1568, 112))
    del arg178_1
    del buf97
    buf99 = buf93; del buf93  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_49(c_void_p(buf99.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()))
    del arg293_1
    del arg294_1
    del arg96_1
    del arg97_1
    del buf98
    # Source Nodes: [shortcut_16, x_268, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf100 = extern_kernels.convolution(buf99, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf100, (8, 672, 14, 14), (131712, 1, 9408, 672))
    del arg179_1
    del buf99
    buf101 = buf100; del buf100  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_50(c_void_p(buf101.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()))
    del arg295_1
    del arg296_1
    del arg98_1
    del arg99_1
    # Source Nodes: [x_274, x_277, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf102 = extern_kernels.convolution(buf101, arg180_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf102, (8, 672, 7, 7), (32928, 1, 4704, 672))
    del arg180_1
    del buf101
    buf103 = buf102; del buf102  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_51(c_void_p(buf103.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    del arg297_1
    del arg298_1
    # Source Nodes: [x_279, x_282, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf104 = extern_kernels.convolution(buf103, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf104, (8, 184, 7, 7), (9016, 1, 1288, 184))
    del arg181_1
    del buf103
    buf105 = buf104; del buf104  # reuse
    cpp_fused__native_batch_norm_legit_no_training_52(c_void_p(buf105.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()))
    del arg102_1
    del arg103_1
    del arg299_1
    del arg300_1
    # Source Nodes: [x_289], Original ATen: [aten.convolution]
    buf106 = extern_kernels.convolution(buf105, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf106, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    del arg182_1
    buf107 = buf106; del buf106  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_53(c_void_p(buf107.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()))
    del arg104_1
    del arg105_1
    del arg301_1
    del arg302_1
    # Source Nodes: [x_290, x_293, x_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf108 = extern_kernels.convolution(buf107, arg183_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
    assert_size_stride(buf108, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    del arg183_1
    del buf107
    buf109 = buf108; del buf108  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_54(c_void_p(buf109.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg303_1
    del arg304_1
    # Source Nodes: [x_295, x_298, x_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf110 = extern_kernels.convolution(buf109, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf110, (8, 184, 7, 7), (9016, 1, 1288, 184))
    del arg184_1
    del buf109
    buf111 = buf105; del buf105  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_55(c_void_p(buf111.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()))
    del arg108_1
    del arg109_1
    del arg305_1
    del arg306_1
    del buf110
    # Source Nodes: [x_306], Original ATen: [aten.convolution]
    buf112 = extern_kernels.convolution(buf111, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf112, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    del arg185_1
    buf113 = buf112; del buf112  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_56(c_void_p(buf113.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()))
    del arg110_1
    del arg111_1
    del arg307_1
    del arg308_1
    # Source Nodes: [x_307, x_310, x_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf114 = extern_kernels.convolution(buf113, arg186_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
    assert_size_stride(buf114, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    del arg186_1
    del buf113
    buf115 = buf114; del buf114  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_57(c_void_p(buf115.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()))
    del arg112_1
    del arg113_1
    del arg309_1
    del arg310_1
    # Source Nodes: [x_312, x_315, x_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf116 = extern_kernels.convolution(buf115, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf116, (8, 184, 7, 7), (9016, 1, 1288, 184))
    del arg187_1
    del buf115
    buf117 = buf111; del buf111  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_58(c_void_p(buf117.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()))
    del arg114_1
    del arg115_1
    del arg311_1
    del arg312_1
    del buf116
    # Source Nodes: [x_323], Original ATen: [aten.convolution]
    buf118 = extern_kernels.convolution(buf117, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf118, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    del arg188_1
    buf119 = buf118; del buf118  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_59(c_void_p(buf119.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()))
    del arg116_1
    del arg117_1
    del arg313_1
    del arg314_1
    # Source Nodes: [x_324, x_327, x_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf120 = extern_kernels.convolution(buf119, arg189_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
    assert_size_stride(buf120, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    del arg189_1
    del buf119
    buf121 = buf120; del buf120  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_60(c_void_p(buf121.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()))
    del arg118_1
    del arg119_1
    del arg315_1
    del arg316_1
    # Source Nodes: [x_329, x_332, x_334], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf122 = extern_kernels.convolution(buf121, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf122, (8, 184, 7, 7), (9016, 1, 1288, 184))
    del arg190_1
    del buf121
    buf123 = buf117; del buf117  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_61(c_void_p(buf123.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()))
    del arg120_1
    del arg121_1
    del arg317_1
    del arg318_1
    del buf122
    # Source Nodes: [shortcut_20, x_335, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf124 = extern_kernels.convolution(buf123, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf124, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    del arg191_1
    del buf123
    buf125 = buf124; del buf124  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_62(c_void_p(buf125.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()))
    del arg122_1
    del arg123_1
    del arg319_1
    del arg320_1
    # Source Nodes: [x_341, x_344, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf126 = extern_kernels.convolution(buf125, arg192_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
    assert_size_stride(buf126, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    del arg192_1
    del buf125
    buf127 = buf126; del buf126  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_63(c_void_p(buf127.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()))
    del arg124_1
    del arg125_1
    del arg321_1
    del arg322_1
    # Source Nodes: [x_346, x_349, x_351], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf128 = extern_kernels.convolution(buf127, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf128, (8, 352, 7, 7), (17248, 1, 2464, 352))
    del arg193_1
    del buf127
    buf129 = buf128; del buf128  # reuse
    cpp_fused__native_batch_norm_legit_no_training_64(c_void_p(buf129.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()))
    del arg126_1
    del arg127_1
    del arg323_1
    del arg324_1
    # Source Nodes: [x_352, x_357], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf130 = extern_kernels.convolution(buf129, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf130, (8, 1984, 7, 7), (97216, 1, 13888, 1984))
    del arg194_1
    del buf129
    buf131 = empty_strided((8, 1984, 1, 1), (1984, 1, 15872, 15872), device='cpu', dtype=torch.float32)
    buf132 = reinterpret_tensor(buf131, (8, 1984, 1, 1), (1984, 1, 1, 1), 0); del buf131  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_65(c_void_p(buf132.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg129_1.data_ptr()))
    del arg128_1
    del arg129_1
    del arg325_1
    del arg326_1
    del buf130
    buf133 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_366], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg196_1, reinterpret_tensor(buf132, (8, 1984), (1984, 1), 0), reinterpret_tensor(arg195_1, (1984, 1000), (1, 1984), 0), alpha=1, beta=1, out=buf133)
    del arg195_1
    del arg196_1
    return (buf133, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((1984, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((1984, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((32, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((96, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((96, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((32, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((192, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((384, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((384, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((384, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((112, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((336, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((112, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((184, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((184, 1104, 1, 1), (1104, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((184, 1104, 1, 1), (1104, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((184, 1104, 1, 1), (1104, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((1104, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((352, 1104, 1, 1), (1104, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((1984, 352, 1, 1), (352, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((1000, 1984), (1984, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((1984, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((1984, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('fbnetc_100', benchmark_compiled_module)
