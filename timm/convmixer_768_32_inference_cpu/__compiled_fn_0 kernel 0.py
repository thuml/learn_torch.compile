
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (147L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (147L*x0))] = tmp0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (786432L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(1e-05);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 + tmp6;
                            auto tmp8 = tmp7.sqrt();
                            auto tmp9 = tmp8.reciprocal();
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp3 * tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp17 = tmp15 + tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp17;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1 = args
    args.clear()
    assert_size_stride(arg0_1, (768, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (768, ), (1, ))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (768, ), (1, ))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (768, ), (1, ))
    assert_size_stride(arg203_1, (768, ), (1, ))
    assert_size_stride(arg204_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg205_1, (768, ), (1, ))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg209_1, (768, ), (1, ))
    assert_size_stride(arg210_1, (768, ), (1, ))
    assert_size_stride(arg211_1, (768, ), (1, ))
    assert_size_stride(arg212_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (768, ), (1, ))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (768, ), (1, ))
    assert_size_stride(arg219_1, (768, ), (1, ))
    assert_size_stride(arg220_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg221_1, (768, ), (1, ))
    assert_size_stride(arg222_1, (768, ), (1, ))
    assert_size_stride(arg223_1, (768, ), (1, ))
    assert_size_stride(arg224_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg225_1, (768, ), (1, ))
    assert_size_stride(arg226_1, (768, ), (1, ))
    assert_size_stride(arg227_1, (768, ), (1, ))
    assert_size_stride(arg228_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg229_1, (768, ), (1, ))
    assert_size_stride(arg230_1, (768, ), (1, ))
    assert_size_stride(arg231_1, (768, ), (1, ))
    assert_size_stride(arg232_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg233_1, (768, ), (1, ))
    assert_size_stride(arg234_1, (768, ), (1, ))
    assert_size_stride(arg235_1, (768, ), (1, ))
    assert_size_stride(arg236_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg237_1, (768, ), (1, ))
    assert_size_stride(arg238_1, (768, ), (1, ))
    assert_size_stride(arg239_1, (768, ), (1, ))
    assert_size_stride(arg240_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg241_1, (768, ), (1, ))
    assert_size_stride(arg242_1, (768, ), (1, ))
    assert_size_stride(arg243_1, (768, ), (1, ))
    assert_size_stride(arg244_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg245_1, (768, ), (1, ))
    assert_size_stride(arg246_1, (768, ), (1, ))
    assert_size_stride(arg247_1, (768, ), (1, ))
    assert_size_stride(arg248_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg249_1, (768, ), (1, ))
    assert_size_stride(arg250_1, (768, ), (1, ))
    assert_size_stride(arg251_1, (768, ), (1, ))
    assert_size_stride(arg252_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg253_1, (768, ), (1, ))
    assert_size_stride(arg254_1, (768, ), (1, ))
    assert_size_stride(arg255_1, (768, ), (1, ))
    assert_size_stride(arg256_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (768, ), (1, ))
    assert_size_stride(arg259_1, (768, ), (1, ))
    assert_size_stride(arg260_1, (1000, 768), (768, 1))
    assert_size_stride(arg261_1, (1000, ), (1, ))
    assert_size_stride(arg262_1, (768, ), (1, ))
    assert_size_stride(arg263_1, (768, ), (1, ))
    assert_size_stride(arg264_1, (), ())
    assert_size_stride(arg265_1, (768, ), (1, ))
    assert_size_stride(arg266_1, (768, ), (1, ))
    assert_size_stride(arg267_1, (), ())
    assert_size_stride(arg268_1, (768, ), (1, ))
    assert_size_stride(arg269_1, (768, ), (1, ))
    assert_size_stride(arg270_1, (), ())
    assert_size_stride(arg271_1, (768, ), (1, ))
    assert_size_stride(arg272_1, (768, ), (1, ))
    assert_size_stride(arg273_1, (), ())
    assert_size_stride(arg274_1, (768, ), (1, ))
    assert_size_stride(arg275_1, (768, ), (1, ))
    assert_size_stride(arg276_1, (), ())
    assert_size_stride(arg277_1, (768, ), (1, ))
    assert_size_stride(arg278_1, (768, ), (1, ))
    assert_size_stride(arg279_1, (), ())
    assert_size_stride(arg280_1, (768, ), (1, ))
    assert_size_stride(arg281_1, (768, ), (1, ))
    assert_size_stride(arg282_1, (), ())
    assert_size_stride(arg283_1, (768, ), (1, ))
    assert_size_stride(arg284_1, (768, ), (1, ))
    assert_size_stride(arg285_1, (), ())
    assert_size_stride(arg286_1, (768, ), (1, ))
    assert_size_stride(arg287_1, (768, ), (1, ))
    assert_size_stride(arg288_1, (), ())
    assert_size_stride(arg289_1, (768, ), (1, ))
    assert_size_stride(arg290_1, (768, ), (1, ))
    assert_size_stride(arg291_1, (), ())
    assert_size_stride(arg292_1, (768, ), (1, ))
    assert_size_stride(arg293_1, (768, ), (1, ))
    assert_size_stride(arg294_1, (), ())
    assert_size_stride(arg295_1, (768, ), (1, ))
    assert_size_stride(arg296_1, (768, ), (1, ))
    assert_size_stride(arg297_1, (), ())
    assert_size_stride(arg298_1, (768, ), (1, ))
    assert_size_stride(arg299_1, (768, ), (1, ))
    assert_size_stride(arg300_1, (), ())
    assert_size_stride(arg301_1, (768, ), (1, ))
    assert_size_stride(arg302_1, (768, ), (1, ))
    assert_size_stride(arg303_1, (), ())
    assert_size_stride(arg304_1, (768, ), (1, ))
    assert_size_stride(arg305_1, (768, ), (1, ))
    assert_size_stride(arg306_1, (), ())
    assert_size_stride(arg307_1, (768, ), (1, ))
    assert_size_stride(arg308_1, (768, ), (1, ))
    assert_size_stride(arg309_1, (), ())
    assert_size_stride(arg310_1, (768, ), (1, ))
    assert_size_stride(arg311_1, (768, ), (1, ))
    assert_size_stride(arg312_1, (), ())
    assert_size_stride(arg313_1, (768, ), (1, ))
    assert_size_stride(arg314_1, (768, ), (1, ))
    assert_size_stride(arg315_1, (), ())
    assert_size_stride(arg316_1, (768, ), (1, ))
    assert_size_stride(arg317_1, (768, ), (1, ))
    assert_size_stride(arg318_1, (), ())
    assert_size_stride(arg319_1, (768, ), (1, ))
    assert_size_stride(arg320_1, (768, ), (1, ))
    assert_size_stride(arg321_1, (), ())
    assert_size_stride(arg322_1, (768, ), (1, ))
    assert_size_stride(arg323_1, (768, ), (1, ))
    assert_size_stride(arg324_1, (), ())
    assert_size_stride(arg325_1, (768, ), (1, ))
    assert_size_stride(arg326_1, (768, ), (1, ))
    assert_size_stride(arg327_1, (), ())
    assert_size_stride(arg328_1, (768, ), (1, ))
    assert_size_stride(arg329_1, (768, ), (1, ))
    assert_size_stride(arg330_1, (), ())
    assert_size_stride(arg331_1, (768, ), (1, ))
    assert_size_stride(arg332_1, (768, ), (1, ))
    assert_size_stride(arg333_1, (), ())
    assert_size_stride(arg334_1, (768, ), (1, ))
    assert_size_stride(arg335_1, (768, ), (1, ))
    assert_size_stride(arg336_1, (), ())
    assert_size_stride(arg337_1, (768, ), (1, ))
    assert_size_stride(arg338_1, (768, ), (1, ))
    assert_size_stride(arg339_1, (), ())
    assert_size_stride(arg340_1, (768, ), (1, ))
    assert_size_stride(arg341_1, (768, ), (1, ))
    assert_size_stride(arg342_1, (), ())
    assert_size_stride(arg343_1, (768, ), (1, ))
    assert_size_stride(arg344_1, (768, ), (1, ))
    assert_size_stride(arg345_1, (), ())
    assert_size_stride(arg346_1, (768, ), (1, ))
    assert_size_stride(arg347_1, (768, ), (1, ))
    assert_size_stride(arg348_1, (), ())
    assert_size_stride(arg349_1, (768, ), (1, ))
    assert_size_stride(arg350_1, (768, ), (1, ))
    assert_size_stride(arg351_1, (), ())
    assert_size_stride(arg352_1, (768, ), (1, ))
    assert_size_stride(arg353_1, (768, ), (1, ))
    assert_size_stride(arg354_1, (), ())
    assert_size_stride(arg355_1, (768, ), (1, ))
    assert_size_stride(arg356_1, (768, ), (1, ))
    assert_size_stride(arg357_1, (), ())
    assert_size_stride(arg358_1, (768, ), (1, ))
    assert_size_stride(arg359_1, (768, ), (1, ))
    assert_size_stride(arg360_1, (), ())
    assert_size_stride(arg361_1, (768, ), (1, ))
    assert_size_stride(arg362_1, (768, ), (1, ))
    assert_size_stride(arg363_1, (), ())
    assert_size_stride(arg364_1, (768, ), (1, ))
    assert_size_stride(arg365_1, (768, ), (1, ))
    assert_size_stride(arg366_1, (), ())
    assert_size_stride(arg367_1, (768, ), (1, ))
    assert_size_stride(arg368_1, (768, ), (1, ))
    assert_size_stride(arg369_1, (), ())
    assert_size_stride(arg370_1, (768, ), (1, ))
    assert_size_stride(arg371_1, (768, ), (1, ))
    assert_size_stride(arg372_1, (), ())
    assert_size_stride(arg373_1, (768, ), (1, ))
    assert_size_stride(arg374_1, (768, ), (1, ))
    assert_size_stride(arg375_1, (), ())
    assert_size_stride(arg376_1, (768, ), (1, ))
    assert_size_stride(arg377_1, (768, ), (1, ))
    assert_size_stride(arg378_1, (), ())
    assert_size_stride(arg379_1, (768, ), (1, ))
    assert_size_stride(arg380_1, (768, ), (1, ))
    assert_size_stride(arg381_1, (), ())
    assert_size_stride(arg382_1, (768, ), (1, ))
    assert_size_stride(arg383_1, (768, ), (1, ))
    assert_size_stride(arg384_1, (), ())
    assert_size_stride(arg385_1, (768, ), (1, ))
    assert_size_stride(arg386_1, (768, ), (1, ))
    assert_size_stride(arg387_1, (), ())
    assert_size_stride(arg388_1, (768, ), (1, ))
    assert_size_stride(arg389_1, (768, ), (1, ))
    assert_size_stride(arg390_1, (), ())
    assert_size_stride(arg391_1, (768, ), (1, ))
    assert_size_stride(arg392_1, (768, ), (1, ))
    assert_size_stride(arg393_1, (), ())
    assert_size_stride(arg394_1, (768, ), (1, ))
    assert_size_stride(arg395_1, (768, ), (1, ))
    assert_size_stride(arg396_1, (), ())
    assert_size_stride(arg397_1, (768, ), (1, ))
    assert_size_stride(arg398_1, (768, ), (1, ))
    assert_size_stride(arg399_1, (), ())
    assert_size_stride(arg400_1, (768, ), (1, ))
    assert_size_stride(arg401_1, (768, ), (1, ))
    assert_size_stride(arg402_1, (), ())
    assert_size_stride(arg403_1, (768, ), (1, ))
    assert_size_stride(arg404_1, (768, ), (1, ))
    assert_size_stride(arg405_1, (), ())
    assert_size_stride(arg406_1, (768, ), (1, ))
    assert_size_stride(arg407_1, (768, ), (1, ))
    assert_size_stride(arg408_1, (), ())
    assert_size_stride(arg409_1, (768, ), (1, ))
    assert_size_stride(arg410_1, (768, ), (1, ))
    assert_size_stride(arg411_1, (), ())
    assert_size_stride(arg412_1, (768, ), (1, ))
    assert_size_stride(arg413_1, (768, ), (1, ))
    assert_size_stride(arg414_1, (), ())
    assert_size_stride(arg415_1, (768, ), (1, ))
    assert_size_stride(arg416_1, (768, ), (1, ))
    assert_size_stride(arg417_1, (), ())
    assert_size_stride(arg418_1, (768, ), (1, ))
    assert_size_stride(arg419_1, (768, ), (1, ))
    assert_size_stride(arg420_1, (), ())
    assert_size_stride(arg421_1, (768, ), (1, ))
    assert_size_stride(arg422_1, (768, ), (1, ))
    assert_size_stride(arg423_1, (), ())
    assert_size_stride(arg424_1, (768, ), (1, ))
    assert_size_stride(arg425_1, (768, ), (1, ))
    assert_size_stride(arg426_1, (), ())
    assert_size_stride(arg427_1, (768, ), (1, ))
    assert_size_stride(arg428_1, (768, ), (1, ))
    assert_size_stride(arg429_1, (), ())
    assert_size_stride(arg430_1, (768, ), (1, ))
    assert_size_stride(arg431_1, (768, ), (1, ))
    assert_size_stride(arg432_1, (), ())
    assert_size_stride(arg433_1, (768, ), (1, ))
    assert_size_stride(arg434_1, (768, ), (1, ))
    assert_size_stride(arg435_1, (), ())
    assert_size_stride(arg436_1, (768, ), (1, ))
    assert_size_stride(arg437_1, (768, ), (1, ))
    assert_size_stride(arg438_1, (), ())
    assert_size_stride(arg439_1, (768, ), (1, ))
    assert_size_stride(arg440_1, (768, ), (1, ))
    assert_size_stride(arg441_1, (), ())
    assert_size_stride(arg442_1, (768, ), (1, ))
    assert_size_stride(arg443_1, (768, ), (1, ))
    assert_size_stride(arg444_1, (), ())
    assert_size_stride(arg445_1, (768, ), (1, ))
    assert_size_stride(arg446_1, (768, ), (1, ))
    assert_size_stride(arg447_1, (), ())
    assert_size_stride(arg448_1, (768, ), (1, ))
    assert_size_stride(arg449_1, (768, ), (1, ))
    assert_size_stride(arg450_1, (), ())
    assert_size_stride(arg451_1, (768, ), (1, ))
    assert_size_stride(arg452_1, (768, ), (1, ))
    assert_size_stride(arg453_1, (), ())
    assert_size_stride(arg454_1, (768, ), (1, ))
    assert_size_stride(arg455_1, (768, ), (1, ))
    assert_size_stride(arg456_1, (), ())
    assert_size_stride(arg457_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((768, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg457_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg457_1
    # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg1_1, stride=(7, 7), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg1_1
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()))
    del arg262_1
    del arg263_1
    del arg2_1
    del arg3_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___fn_0], Original ATen: [aten.convolution]
    buf4 = extern_kernels.convolution(buf3, arg4_1, arg5_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf4, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg4_1
    del arg5_1
    buf5 = buf3; del buf3  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_2(c_void_p(buf5.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()))
    del arg265_1
    del arg266_1
    del arg6_1
    del arg7_1
    del buf4
    # Source Nodes: [add, getattr_getattr_l__mod___blocks___0_____0___fn_1, getattr_getattr_l__mod___blocks___0_____0___fn_2, l__mod___blocks_0_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf6 = extern_kernels.convolution(buf5, arg8_1, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf6, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg8_1
    del arg9_1
    del buf5
    buf7 = buf6; del buf6  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_3(c_void_p(buf7.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()))
    del arg10_1
    del arg11_1
    del arg268_1
    del arg269_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___fn_0], Original ATen: [aten.convolution]
    buf8 = extern_kernels.convolution(buf7, arg12_1, arg13_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf8, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg12_1
    del arg13_1
    buf9 = buf7; del buf7  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_4(c_void_p(buf9.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()))
    del arg14_1
    del arg15_1
    del arg271_1
    del arg272_1
    del buf8
    # Source Nodes: [add_1, getattr_getattr_l__mod___blocks___1_____0___fn_1, getattr_getattr_l__mod___blocks___1_____0___fn_2, l__mod___blocks_1_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf10 = extern_kernels.convolution(buf9, arg16_1, arg17_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf10, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg16_1
    del arg17_1
    del buf9
    buf11 = buf10; del buf10  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_5(c_void_p(buf11.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()))
    del arg18_1
    del arg19_1
    del arg274_1
    del arg275_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___fn_0], Original ATen: [aten.convolution]
    buf12 = extern_kernels.convolution(buf11, arg20_1, arg21_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf12, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg20_1
    del arg21_1
    buf13 = buf11; del buf11  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_6(c_void_p(buf13.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg22_1
    del arg23_1
    del arg277_1
    del arg278_1
    del buf12
    # Source Nodes: [add_2, getattr_getattr_l__mod___blocks___2_____0___fn_1, getattr_getattr_l__mod___blocks___2_____0___fn_2, l__mod___blocks_2_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf14 = extern_kernels.convolution(buf13, arg24_1, arg25_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf14, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg24_1
    del arg25_1
    del buf13
    buf15 = buf14; del buf14  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_7(c_void_p(buf15.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()))
    del arg26_1
    del arg27_1
    del arg280_1
    del arg281_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___fn_0], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf15, arg28_1, arg29_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf16, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg28_1
    del arg29_1
    buf17 = buf15; del buf15  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_8(c_void_p(buf17.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()))
    del arg283_1
    del arg284_1
    del arg30_1
    del arg31_1
    del buf16
    # Source Nodes: [add_3, getattr_getattr_l__mod___blocks___3_____0___fn_1, getattr_getattr_l__mod___blocks___3_____0___fn_2, l__mod___blocks_3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf18 = extern_kernels.convolution(buf17, arg32_1, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf18, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg32_1
    del arg33_1
    del buf17
    buf19 = buf18; del buf18  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_9(c_void_p(buf19.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg286_1
    del arg287_1
    del arg34_1
    del arg35_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___fn_0], Original ATen: [aten.convolution]
    buf20 = extern_kernels.convolution(buf19, arg36_1, arg37_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf20, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg36_1
    del arg37_1
    buf21 = buf19; del buf19  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_10(c_void_p(buf21.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()))
    del arg289_1
    del arg290_1
    del arg38_1
    del arg39_1
    del buf20
    # Source Nodes: [add_4, getattr_getattr_l__mod___blocks___4_____0___fn_1, getattr_getattr_l__mod___blocks___4_____0___fn_2, l__mod___blocks_4_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf22 = extern_kernels.convolution(buf21, arg40_1, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf22, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg40_1
    del arg41_1
    del buf21
    buf23 = buf22; del buf22  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_11(c_void_p(buf23.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()))
    del arg292_1
    del arg293_1
    del arg42_1
    del arg43_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___fn_0], Original ATen: [aten.convolution]
    buf24 = extern_kernels.convolution(buf23, arg44_1, arg45_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf24, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg44_1
    del arg45_1
    buf25 = buf23; del buf23  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_12(c_void_p(buf25.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg295_1
    del arg296_1
    del arg46_1
    del arg47_1
    del buf24
    # Source Nodes: [add_5, getattr_getattr_l__mod___blocks___5_____0___fn_1, getattr_getattr_l__mod___blocks___5_____0___fn_2, l__mod___blocks_5_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf26 = extern_kernels.convolution(buf25, arg48_1, arg49_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf26, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg48_1
    del arg49_1
    del buf25
    buf27 = buf26; del buf26  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_13(c_void_p(buf27.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()))
    del arg298_1
    del arg299_1
    del arg50_1
    del arg51_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___fn_0], Original ATen: [aten.convolution]
    buf28 = extern_kernels.convolution(buf27, arg52_1, arg53_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf28, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg52_1
    del arg53_1
    buf29 = buf27; del buf27  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_14(c_void_p(buf29.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()))
    del arg301_1
    del arg302_1
    del arg54_1
    del arg55_1
    del buf28
    # Source Nodes: [add_6, getattr_getattr_l__mod___blocks___6_____0___fn_1, getattr_getattr_l__mod___blocks___6_____0___fn_2, l__mod___blocks_6_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf30 = extern_kernels.convolution(buf29, arg56_1, arg57_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf30, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg56_1
    del arg57_1
    del buf29
    buf31 = buf30; del buf30  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_15(c_void_p(buf31.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg304_1
    del arg305_1
    del arg58_1
    del arg59_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___fn_0], Original ATen: [aten.convolution]
    buf32 = extern_kernels.convolution(buf31, arg60_1, arg61_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf32, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg60_1
    del arg61_1
    buf33 = buf31; del buf31  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_16(c_void_p(buf33.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()))
    del arg307_1
    del arg308_1
    del arg62_1
    del arg63_1
    del buf32
    # Source Nodes: [add_7, getattr_getattr_l__mod___blocks___7_____0___fn_1, getattr_getattr_l__mod___blocks___7_____0___fn_2, l__mod___blocks_7_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf34 = extern_kernels.convolution(buf33, arg64_1, arg65_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf34, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg64_1
    del arg65_1
    del buf33
    buf35 = buf34; del buf34  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_17(c_void_p(buf35.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg310_1
    del arg311_1
    del arg66_1
    del arg67_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___fn_0], Original ATen: [aten.convolution]
    buf36 = extern_kernels.convolution(buf35, arg68_1, arg69_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf36, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg68_1
    del arg69_1
    buf37 = buf35; del buf35  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_18(c_void_p(buf37.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg313_1
    del arg314_1
    del arg70_1
    del arg71_1
    del buf36
    # Source Nodes: [add_8, getattr_getattr_l__mod___blocks___8_____0___fn_1, getattr_getattr_l__mod___blocks___8_____0___fn_2, l__mod___blocks_8_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf38 = extern_kernels.convolution(buf37, arg72_1, arg73_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf38, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg72_1
    del arg73_1
    del buf37
    buf39 = buf38; del buf38  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_19(c_void_p(buf39.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()))
    del arg316_1
    del arg317_1
    del arg74_1
    del arg75_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___9_____0___fn_0], Original ATen: [aten.convolution]
    buf40 = extern_kernels.convolution(buf39, arg76_1, arg77_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf40, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg76_1
    del arg77_1
    buf41 = buf39; del buf39  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_20(c_void_p(buf41.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()))
    del arg319_1
    del arg320_1
    del arg78_1
    del arg79_1
    del buf40
    # Source Nodes: [add_9, getattr_getattr_l__mod___blocks___9_____0___fn_1, getattr_getattr_l__mod___blocks___9_____0___fn_2, l__mod___blocks_9_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf42 = extern_kernels.convolution(buf41, arg80_1, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf42, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg80_1
    del arg81_1
    del buf41
    buf43 = buf42; del buf42  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_21(c_void_p(buf43.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg322_1
    del arg323_1
    del arg82_1
    del arg83_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___10_____0___fn_0], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf43, arg84_1, arg85_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf44, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg84_1
    del arg85_1
    buf45 = buf43; del buf43  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_22(c_void_p(buf45.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()))
    del arg325_1
    del arg326_1
    del arg86_1
    del arg87_1
    del buf44
    # Source Nodes: [add_10, getattr_getattr_l__mod___blocks___10_____0___fn_1, getattr_getattr_l__mod___blocks___10_____0___fn_2, l__mod___blocks_10_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf46 = extern_kernels.convolution(buf45, arg88_1, arg89_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf46, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg88_1
    del arg89_1
    del buf45
    buf47 = buf46; del buf46  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_23(c_void_p(buf47.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()))
    del arg328_1
    del arg329_1
    del arg90_1
    del arg91_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___11_____0___fn_0], Original ATen: [aten.convolution]
    buf48 = extern_kernels.convolution(buf47, arg92_1, arg93_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf48, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg92_1
    del arg93_1
    buf49 = buf47; del buf47  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_24(c_void_p(buf49.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg331_1
    del arg332_1
    del arg94_1
    del arg95_1
    del buf48
    # Source Nodes: [add_11, getattr_getattr_l__mod___blocks___11_____0___fn_1, getattr_getattr_l__mod___blocks___11_____0___fn_2, l__mod___blocks_11_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf50 = extern_kernels.convolution(buf49, arg96_1, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf50, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg96_1
    del arg97_1
    del buf49
    buf51 = buf50; del buf50  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_25(c_void_p(buf51.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()))
    del arg334_1
    del arg335_1
    del arg98_1
    del arg99_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___12_____0___fn_0], Original ATen: [aten.convolution]
    buf52 = extern_kernels.convolution(buf51, arg100_1, arg101_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf52, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg100_1
    del arg101_1
    buf53 = buf51; del buf51  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_26(c_void_p(buf53.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()))
    del arg102_1
    del arg103_1
    del arg337_1
    del arg338_1
    del buf52
    # Source Nodes: [add_12, getattr_getattr_l__mod___blocks___12_____0___fn_1, getattr_getattr_l__mod___blocks___12_____0___fn_2, l__mod___blocks_12_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf54 = extern_kernels.convolution(buf53, arg104_1, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf54, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg104_1
    del arg105_1
    del buf53
    buf55 = buf54; del buf54  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_27(c_void_p(buf55.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg340_1
    del arg341_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___13_____0___fn_0], Original ATen: [aten.convolution]
    buf56 = extern_kernels.convolution(buf55, arg108_1, arg109_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf56, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg108_1
    del arg109_1
    buf57 = buf55; del buf55  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_28(c_void_p(buf57.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()))
    del arg110_1
    del arg111_1
    del arg343_1
    del arg344_1
    del buf56
    # Source Nodes: [add_13, getattr_getattr_l__mod___blocks___13_____0___fn_1, getattr_getattr_l__mod___blocks___13_____0___fn_2, l__mod___blocks_13_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf58 = extern_kernels.convolution(buf57, arg112_1, arg113_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf58, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg112_1
    del arg113_1
    del buf57
    buf59 = buf58; del buf58  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_29(c_void_p(buf59.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()))
    del arg114_1
    del arg115_1
    del arg346_1
    del arg347_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___14_____0___fn_0], Original ATen: [aten.convolution]
    buf60 = extern_kernels.convolution(buf59, arg116_1, arg117_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf60, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg116_1
    del arg117_1
    buf61 = buf59; del buf59  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_30(c_void_p(buf61.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()))
    del arg118_1
    del arg119_1
    del arg349_1
    del arg350_1
    del buf60
    # Source Nodes: [add_14, getattr_getattr_l__mod___blocks___14_____0___fn_1, getattr_getattr_l__mod___blocks___14_____0___fn_2, l__mod___blocks_14_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf62 = extern_kernels.convolution(buf61, arg120_1, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf62, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg120_1
    del arg121_1
    del buf61
    buf63 = buf62; del buf62  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_31(c_void_p(buf63.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()))
    del arg122_1
    del arg123_1
    del arg352_1
    del arg353_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___15_____0___fn_0], Original ATen: [aten.convolution]
    buf64 = extern_kernels.convolution(buf63, arg124_1, arg125_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf64, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg124_1
    del arg125_1
    buf65 = buf63; del buf63  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_32(c_void_p(buf65.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()))
    del arg126_1
    del arg127_1
    del arg355_1
    del arg356_1
    del buf64
    # Source Nodes: [add_15, getattr_getattr_l__mod___blocks___15_____0___fn_1, getattr_getattr_l__mod___blocks___15_____0___fn_2, l__mod___blocks_15_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf66 = extern_kernels.convolution(buf65, arg128_1, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf66, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg128_1
    del arg129_1
    del buf65
    buf67 = buf66; del buf66  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_33(c_void_p(buf67.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()))
    del arg130_1
    del arg131_1
    del arg358_1
    del arg359_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___16_____0___fn_0], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, arg132_1, arg133_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf68, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg132_1
    del arg133_1
    buf69 = buf67; del buf67  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_34(c_void_p(buf69.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg135_1.data_ptr()))
    del arg134_1
    del arg135_1
    del arg361_1
    del arg362_1
    del buf68
    # Source Nodes: [add_16, getattr_getattr_l__mod___blocks___16_____0___fn_1, getattr_getattr_l__mod___blocks___16_____0___fn_2, l__mod___blocks_16_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf70 = extern_kernels.convolution(buf69, arg136_1, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf70, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg136_1
    del arg137_1
    del buf69
    buf71 = buf70; del buf70  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_35(c_void_p(buf71.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()))
    del arg138_1
    del arg139_1
    del arg364_1
    del arg365_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___17_____0___fn_0], Original ATen: [aten.convolution]
    buf72 = extern_kernels.convolution(buf71, arg140_1, arg141_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf72, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg140_1
    del arg141_1
    buf73 = buf71; del buf71  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_36(c_void_p(buf73.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()))
    del arg142_1
    del arg143_1
    del arg367_1
    del arg368_1
    del buf72
    # Source Nodes: [add_17, getattr_getattr_l__mod___blocks___17_____0___fn_1, getattr_getattr_l__mod___blocks___17_____0___fn_2, l__mod___blocks_17_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf74 = extern_kernels.convolution(buf73, arg144_1, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf74, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg144_1
    del arg145_1
    del buf73
    buf75 = buf74; del buf74  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_37(c_void_p(buf75.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()))
    del arg146_1
    del arg147_1
    del arg370_1
    del arg371_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___18_____0___fn_0], Original ATen: [aten.convolution]
    buf76 = extern_kernels.convolution(buf75, arg148_1, arg149_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf76, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg148_1
    del arg149_1
    buf77 = buf75; del buf75  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_38(c_void_p(buf77.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(arg373_1.data_ptr()), c_void_p(arg374_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(arg151_1.data_ptr()))
    del arg150_1
    del arg151_1
    del arg373_1
    del arg374_1
    del buf76
    # Source Nodes: [add_18, getattr_getattr_l__mod___blocks___18_____0___fn_1, getattr_getattr_l__mod___blocks___18_____0___fn_2, l__mod___blocks_18_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf78 = extern_kernels.convolution(buf77, arg152_1, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf78, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg152_1
    del arg153_1
    del buf77
    buf79 = buf78; del buf78  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_39(c_void_p(buf79.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(arg377_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()))
    del arg154_1
    del arg155_1
    del arg376_1
    del arg377_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___19_____0___fn_0], Original ATen: [aten.convolution]
    buf80 = extern_kernels.convolution(buf79, arg156_1, arg157_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf80, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg156_1
    del arg157_1
    buf81 = buf79; del buf79  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_40(c_void_p(buf81.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg159_1.data_ptr()))
    del arg158_1
    del arg159_1
    del arg379_1
    del arg380_1
    del buf80
    # Source Nodes: [add_19, getattr_getattr_l__mod___blocks___19_____0___fn_1, getattr_getattr_l__mod___blocks___19_____0___fn_2, l__mod___blocks_19_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf82 = extern_kernels.convolution(buf81, arg160_1, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf82, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg160_1
    del arg161_1
    del buf81
    buf83 = buf82; del buf82  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_41(c_void_p(buf83.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(arg383_1.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()))
    del arg162_1
    del arg163_1
    del arg382_1
    del arg383_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___20_____0___fn_0], Original ATen: [aten.convolution]
    buf84 = extern_kernels.convolution(buf83, arg164_1, arg165_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf84, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg164_1
    del arg165_1
    buf85 = buf83; del buf83  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_42(c_void_p(buf85.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg167_1.data_ptr()))
    del arg166_1
    del arg167_1
    del arg385_1
    del arg386_1
    del buf84
    # Source Nodes: [add_20, getattr_getattr_l__mod___blocks___20_____0___fn_1, getattr_getattr_l__mod___blocks___20_____0___fn_2, l__mod___blocks_20_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf86 = extern_kernels.convolution(buf85, arg168_1, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf86, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg168_1
    del arg169_1
    del buf85
    buf87 = buf86; del buf86  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_43(c_void_p(buf87.data_ptr()), c_void_p(arg388_1.data_ptr()), c_void_p(arg389_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg171_1.data_ptr()))
    del arg170_1
    del arg171_1
    del arg388_1
    del arg389_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___21_____0___fn_0], Original ATen: [aten.convolution]
    buf88 = extern_kernels.convolution(buf87, arg172_1, arg173_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf88, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg172_1
    del arg173_1
    buf89 = buf87; del buf87  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_44(c_void_p(buf89.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(arg391_1.data_ptr()), c_void_p(arg392_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()))
    del arg174_1
    del arg175_1
    del arg391_1
    del arg392_1
    del buf88
    # Source Nodes: [add_21, getattr_getattr_l__mod___blocks___21_____0___fn_1, getattr_getattr_l__mod___blocks___21_____0___fn_2, l__mod___blocks_21_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf90 = extern_kernels.convolution(buf89, arg176_1, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf90, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg176_1
    del arg177_1
    del buf89
    buf91 = buf90; del buf90  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_45(c_void_p(buf91.data_ptr()), c_void_p(arg394_1.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()))
    del arg178_1
    del arg179_1
    del arg394_1
    del arg395_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___22_____0___fn_0], Original ATen: [aten.convolution]
    buf92 = extern_kernels.convolution(buf91, arg180_1, arg181_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf92, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg180_1
    del arg181_1
    buf93 = buf91; del buf91  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_46(c_void_p(buf93.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(arg397_1.data_ptr()), c_void_p(arg398_1.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg183_1.data_ptr()))
    del arg182_1
    del arg183_1
    del arg397_1
    del arg398_1
    del buf92
    # Source Nodes: [add_22, getattr_getattr_l__mod___blocks___22_____0___fn_1, getattr_getattr_l__mod___blocks___22_____0___fn_2, l__mod___blocks_22_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf94 = extern_kernels.convolution(buf93, arg184_1, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf94, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg184_1
    del arg185_1
    del buf93
    buf95 = buf94; del buf94  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_47(c_void_p(buf95.data_ptr()), c_void_p(arg400_1.data_ptr()), c_void_p(arg401_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()))
    del arg186_1
    del arg187_1
    del arg400_1
    del arg401_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___23_____0___fn_0], Original ATen: [aten.convolution]
    buf96 = extern_kernels.convolution(buf95, arg188_1, arg189_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf96, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg188_1
    del arg189_1
    buf97 = buf95; del buf95  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_48(c_void_p(buf97.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(arg403_1.data_ptr()), c_void_p(arg404_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg191_1.data_ptr()))
    del arg190_1
    del arg191_1
    del arg403_1
    del arg404_1
    del buf96
    # Source Nodes: [add_23, getattr_getattr_l__mod___blocks___23_____0___fn_1, getattr_getattr_l__mod___blocks___23_____0___fn_2, l__mod___blocks_23_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf98 = extern_kernels.convolution(buf97, arg192_1, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf98, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg192_1
    del arg193_1
    del buf97
    buf99 = buf98; del buf98  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_49(c_void_p(buf99.data_ptr()), c_void_p(arg406_1.data_ptr()), c_void_p(arg407_1.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()))
    del arg194_1
    del arg195_1
    del arg406_1
    del arg407_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___24_____0___fn_0], Original ATen: [aten.convolution]
    buf100 = extern_kernels.convolution(buf99, arg196_1, arg197_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf100, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg196_1
    del arg197_1
    buf101 = buf100; del buf100  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_50(c_void_p(buf101.data_ptr()), c_void_p(arg409_1.data_ptr()), c_void_p(arg410_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(buf99.data_ptr()))
    del arg198_1
    del arg199_1
    del arg409_1
    del arg410_1
    del buf99
    # Source Nodes: [add_24, getattr_getattr_l__mod___blocks___24_____0___fn_1, getattr_getattr_l__mod___blocks___24_____0___fn_2, l__mod___blocks_24_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf102 = extern_kernels.convolution(buf101, arg200_1, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf102, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg200_1
    del arg201_1
    del buf101
    buf103 = buf102; del buf102  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_51(c_void_p(buf103.data_ptr()), c_void_p(arg412_1.data_ptr()), c_void_p(arg413_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()))
    del arg202_1
    del arg203_1
    del arg412_1
    del arg413_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___25_____0___fn_0], Original ATen: [aten.convolution]
    buf104 = extern_kernels.convolution(buf103, arg204_1, arg205_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf104, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg204_1
    del arg205_1
    buf105 = buf103; del buf103  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_52(c_void_p(buf105.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(arg415_1.data_ptr()), c_void_p(arg416_1.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()))
    del arg206_1
    del arg207_1
    del arg415_1
    del arg416_1
    del buf104
    # Source Nodes: [add_25, getattr_getattr_l__mod___blocks___25_____0___fn_1, getattr_getattr_l__mod___blocks___25_____0___fn_2, l__mod___blocks_25_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf106 = extern_kernels.convolution(buf105, arg208_1, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf106, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg208_1
    del arg209_1
    del buf105
    buf107 = buf106; del buf106  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_53(c_void_p(buf107.data_ptr()), c_void_p(arg418_1.data_ptr()), c_void_p(arg419_1.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()))
    del arg210_1
    del arg211_1
    del arg418_1
    del arg419_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___26_____0___fn_0], Original ATen: [aten.convolution]
    buf108 = extern_kernels.convolution(buf107, arg212_1, arg213_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf108, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg212_1
    del arg213_1
    buf109 = buf107; del buf107  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_54(c_void_p(buf109.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(arg421_1.data_ptr()), c_void_p(arg422_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg215_1.data_ptr()))
    del arg214_1
    del arg215_1
    del arg421_1
    del arg422_1
    del buf108
    # Source Nodes: [add_26, getattr_getattr_l__mod___blocks___26_____0___fn_1, getattr_getattr_l__mod___blocks___26_____0___fn_2, l__mod___blocks_26_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf110 = extern_kernels.convolution(buf109, arg216_1, arg217_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf110, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg216_1
    del arg217_1
    del buf109
    buf111 = buf110; del buf110  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_55(c_void_p(buf111.data_ptr()), c_void_p(arg424_1.data_ptr()), c_void_p(arg425_1.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg219_1.data_ptr()))
    del arg218_1
    del arg219_1
    del arg424_1
    del arg425_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___27_____0___fn_0], Original ATen: [aten.convolution]
    buf112 = extern_kernels.convolution(buf111, arg220_1, arg221_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf112, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg220_1
    del arg221_1
    buf113 = buf111; del buf111  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_56(c_void_p(buf113.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(arg427_1.data_ptr()), c_void_p(arg428_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg223_1.data_ptr()))
    del arg222_1
    del arg223_1
    del arg427_1
    del arg428_1
    del buf112
    # Source Nodes: [add_27, getattr_getattr_l__mod___blocks___27_____0___fn_1, getattr_getattr_l__mod___blocks___27_____0___fn_2, l__mod___blocks_27_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf114 = extern_kernels.convolution(buf113, arg224_1, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf114, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg224_1
    del arg225_1
    del buf113
    buf115 = buf114; del buf114  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_57(c_void_p(buf115.data_ptr()), c_void_p(arg430_1.data_ptr()), c_void_p(arg431_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg227_1.data_ptr()))
    del arg226_1
    del arg227_1
    del arg430_1
    del arg431_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___28_____0___fn_0], Original ATen: [aten.convolution]
    buf116 = extern_kernels.convolution(buf115, arg228_1, arg229_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf116, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg228_1
    del arg229_1
    buf117 = buf115; del buf115  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_58(c_void_p(buf117.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(arg433_1.data_ptr()), c_void_p(arg434_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg231_1.data_ptr()))
    del arg230_1
    del arg231_1
    del arg433_1
    del arg434_1
    del buf116
    # Source Nodes: [add_28, getattr_getattr_l__mod___blocks___28_____0___fn_1, getattr_getattr_l__mod___blocks___28_____0___fn_2, l__mod___blocks_28_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf118 = extern_kernels.convolution(buf117, arg232_1, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf118, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg232_1
    del arg233_1
    del buf117
    buf119 = buf118; del buf118  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_59(c_void_p(buf119.data_ptr()), c_void_p(arg436_1.data_ptr()), c_void_p(arg437_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg235_1.data_ptr()))
    del arg234_1
    del arg235_1
    del arg436_1
    del arg437_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___29_____0___fn_0], Original ATen: [aten.convolution]
    buf120 = extern_kernels.convolution(buf119, arg236_1, arg237_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf120, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg236_1
    del arg237_1
    buf121 = buf119; del buf119  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_60(c_void_p(buf121.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(arg439_1.data_ptr()), c_void_p(arg440_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg239_1.data_ptr()))
    del arg238_1
    del arg239_1
    del arg439_1
    del arg440_1
    del buf120
    # Source Nodes: [add_29, getattr_getattr_l__mod___blocks___29_____0___fn_1, getattr_getattr_l__mod___blocks___29_____0___fn_2, l__mod___blocks_29_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf122 = extern_kernels.convolution(buf121, arg240_1, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf122, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg240_1
    del arg241_1
    del buf121
    buf123 = buf122; del buf122  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_61(c_void_p(buf123.data_ptr()), c_void_p(arg442_1.data_ptr()), c_void_p(arg443_1.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()))
    del arg242_1
    del arg243_1
    del arg442_1
    del arg443_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___30_____0___fn_0], Original ATen: [aten.convolution]
    buf124 = extern_kernels.convolution(buf123, arg244_1, arg245_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf124, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg244_1
    del arg245_1
    buf125 = buf123; del buf123  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_62(c_void_p(buf125.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(arg445_1.data_ptr()), c_void_p(arg446_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()))
    del arg246_1
    del arg247_1
    del arg445_1
    del arg446_1
    del buf124
    # Source Nodes: [add_30, getattr_getattr_l__mod___blocks___30_____0___fn_1, getattr_getattr_l__mod___blocks___30_____0___fn_2, l__mod___blocks_30_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf126 = extern_kernels.convolution(buf125, arg248_1, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf126, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg248_1
    del arg249_1
    del buf125
    buf127 = buf126; del buf126  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_63(c_void_p(buf127.data_ptr()), c_void_p(arg448_1.data_ptr()), c_void_p(arg449_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg251_1.data_ptr()))
    del arg250_1
    del arg251_1
    del arg448_1
    del arg449_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_0], Original ATen: [aten.convolution]
    buf128 = extern_kernels.convolution(buf127, arg252_1, arg253_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf128, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg252_1
    del arg253_1
    buf129 = buf127; del buf127  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_64(c_void_p(buf129.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(arg451_1.data_ptr()), c_void_p(arg452_1.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg255_1.data_ptr()))
    del arg254_1
    del arg255_1
    del arg451_1
    del arg452_1
    del buf128
    # Source Nodes: [add_31, getattr_getattr_l__mod___blocks___31_____0___fn_1, getattr_getattr_l__mod___blocks___31_____0___fn_2, l__mod___blocks_31_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf130 = extern_kernels.convolution(buf129, arg256_1, arg257_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf130, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg256_1
    del arg257_1
    del buf129
    buf131 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cpu', dtype=torch.float32)
    buf132 = reinterpret_tensor(buf131, (8, 768, 1, 1), (768, 1, 1, 1), 0); del buf131  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_65(c_void_p(buf132.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(arg454_1.data_ptr()), c_void_p(arg455_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()))
    del arg258_1
    del arg259_1
    del arg454_1
    del arg455_1
    del buf130
    buf133 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg261_1, reinterpret_tensor(buf132, (8, 768), (768, 1), 0), reinterpret_tensor(arg260_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf133)
    del arg260_1
    del arg261_1
    return (buf133, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((768, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg265_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg268_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg271_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg274_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg277_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg280_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg283_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg286_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg289_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg292_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg295_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg298_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg301_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg304_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg307_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg310_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg313_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg316_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg319_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg322_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg325_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg328_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg331_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg334_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg337_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg340_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg343_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg346_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg349_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg352_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg355_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg358_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg361_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg364_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg367_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg370_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg373_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg376_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg379_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg382_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg385_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg388_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg391_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg394_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg397_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg400_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg403_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg406_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg409_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg412_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg415_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg416_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg418_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg419_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg421_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg422_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg424_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg425_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg427_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg428_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg430_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg431_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg433_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg434_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg436_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg437_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg439_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg440_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg442_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg443_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg445_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg446_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg448_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg449_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg451_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg452_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg454_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg455_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg457_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convmixer_768_32', benchmark_compiled_module)
