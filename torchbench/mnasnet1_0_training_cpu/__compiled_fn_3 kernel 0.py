
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


cpp_fused_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
                }
            }
        }
    }
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
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr1[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
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
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
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
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
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
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (24L*x0)));
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
                    tmp18.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
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
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (24L*x0)));
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
                    tmp18.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
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
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
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
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (40L*x0)));
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
                    tmp18.store(out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
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
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (40L*x0)));
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
                    tmp18.store(out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
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
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
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
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (80L*x0)));
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
                tmp18.store(out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
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
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (80L*x0)));
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
                tmp18.store(out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (96L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (96L*x0)));
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
                tmp18.store(out_ptr0 + static_cast<long>(x1 + (96L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (576L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
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
                tmp18.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
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
                tmp18.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
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
                tmp18.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1280L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1280L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1280L*x2) + (62720L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (1280L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5120L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
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


cpp_fused_threshold_backward_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(250880L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp2;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (48, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_11, (48, ), (1, ))
    assert_size_stride(primals_12, (48, ), (1, ))
    assert_size_stride(primals_13, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (48, ), (1, ))
    assert_size_stride(primals_15, (48, ), (1, ))
    assert_size_stride(primals_16, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_19, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_20, (72, ), (1, ))
    assert_size_stride(primals_21, (72, ), (1, ))
    assert_size_stride(primals_22, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (72, ), (1, ))
    assert_size_stride(primals_24, (72, ), (1, ))
    assert_size_stride(primals_25, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_26, (24, ), (1, ))
    assert_size_stride(primals_27, (24, ), (1, ))
    assert_size_stride(primals_28, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_29, (72, ), (1, ))
    assert_size_stride(primals_30, (72, ), (1, ))
    assert_size_stride(primals_31, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (72, ), (1, ))
    assert_size_stride(primals_33, (72, ), (1, ))
    assert_size_stride(primals_34, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_35, (24, ), (1, ))
    assert_size_stride(primals_36, (24, ), (1, ))
    assert_size_stride(primals_37, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_38, (72, ), (1, ))
    assert_size_stride(primals_39, (72, ), (1, ))
    assert_size_stride(primals_40, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_41, (72, ), (1, ))
    assert_size_stride(primals_42, (72, ), (1, ))
    assert_size_stride(primals_43, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_44, (40, ), (1, ))
    assert_size_stride(primals_45, (40, ), (1, ))
    assert_size_stride(primals_46, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_47, (120, ), (1, ))
    assert_size_stride(primals_48, (120, ), (1, ))
    assert_size_stride(primals_49, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_50, (120, ), (1, ))
    assert_size_stride(primals_51, (120, ), (1, ))
    assert_size_stride(primals_52, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_53, (40, ), (1, ))
    assert_size_stride(primals_54, (40, ), (1, ))
    assert_size_stride(primals_55, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_56, (120, ), (1, ))
    assert_size_stride(primals_57, (120, ), (1, ))
    assert_size_stride(primals_58, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_59, (120, ), (1, ))
    assert_size_stride(primals_60, (120, ), (1, ))
    assert_size_stride(primals_61, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_62, (40, ), (1, ))
    assert_size_stride(primals_63, (40, ), (1, ))
    assert_size_stride(primals_64, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_65, (240, ), (1, ))
    assert_size_stride(primals_66, (240, ), (1, ))
    assert_size_stride(primals_67, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_68, (240, ), (1, ))
    assert_size_stride(primals_69, (240, ), (1, ))
    assert_size_stride(primals_70, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_71, (80, ), (1, ))
    assert_size_stride(primals_72, (80, ), (1, ))
    assert_size_stride(primals_73, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_74, (480, ), (1, ))
    assert_size_stride(primals_75, (480, ), (1, ))
    assert_size_stride(primals_76, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_77, (480, ), (1, ))
    assert_size_stride(primals_78, (480, ), (1, ))
    assert_size_stride(primals_79, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_80, (80, ), (1, ))
    assert_size_stride(primals_81, (80, ), (1, ))
    assert_size_stride(primals_82, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_83, (480, ), (1, ))
    assert_size_stride(primals_84, (480, ), (1, ))
    assert_size_stride(primals_85, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_86, (480, ), (1, ))
    assert_size_stride(primals_87, (480, ), (1, ))
    assert_size_stride(primals_88, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_89, (80, ), (1, ))
    assert_size_stride(primals_90, (80, ), (1, ))
    assert_size_stride(primals_91, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_92, (480, ), (1, ))
    assert_size_stride(primals_93, (480, ), (1, ))
    assert_size_stride(primals_94, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_95, (480, ), (1, ))
    assert_size_stride(primals_96, (480, ), (1, ))
    assert_size_stride(primals_97, (96, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_98, (96, ), (1, ))
    assert_size_stride(primals_99, (96, ), (1, ))
    assert_size_stride(primals_100, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_101, (576, ), (1, ))
    assert_size_stride(primals_102, (576, ), (1, ))
    assert_size_stride(primals_103, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_104, (576, ), (1, ))
    assert_size_stride(primals_105, (576, ), (1, ))
    assert_size_stride(primals_106, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_107, (96, ), (1, ))
    assert_size_stride(primals_108, (96, ), (1, ))
    assert_size_stride(primals_109, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_110, (576, ), (1, ))
    assert_size_stride(primals_111, (576, ), (1, ))
    assert_size_stride(primals_112, (576, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_113, (576, ), (1, ))
    assert_size_stride(primals_114, (576, ), (1, ))
    assert_size_stride(primals_115, (192, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_116, (192, ), (1, ))
    assert_size_stride(primals_117, (192, ), (1, ))
    assert_size_stride(primals_118, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_119, (1152, ), (1, ))
    assert_size_stride(primals_120, (1152, ), (1, ))
    assert_size_stride(primals_121, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_122, (1152, ), (1, ))
    assert_size_stride(primals_123, (1152, ), (1, ))
    assert_size_stride(primals_124, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_125, (192, ), (1, ))
    assert_size_stride(primals_126, (192, ), (1, ))
    assert_size_stride(primals_127, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_128, (1152, ), (1, ))
    assert_size_stride(primals_129, (1152, ), (1, ))
    assert_size_stride(primals_130, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_131, (1152, ), (1, ))
    assert_size_stride(primals_132, (1152, ), (1, ))
    assert_size_stride(primals_133, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_135, (192, ), (1, ))
    assert_size_stride(primals_136, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_137, (1152, ), (1, ))
    assert_size_stride(primals_138, (1152, ), (1, ))
    assert_size_stride(primals_139, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_140, (1152, ), (1, ))
    assert_size_stride(primals_141, (1152, ), (1, ))
    assert_size_stride(primals_142, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_143, (192, ), (1, ))
    assert_size_stride(primals_144, (192, ), (1, ))
    assert_size_stride(primals_145, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_146, (1152, ), (1, ))
    assert_size_stride(primals_147, (1152, ), (1, ))
    assert_size_stride(primals_148, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (1152, ), (1, ))
    assert_size_stride(primals_150, (1152, ), (1, ))
    assert_size_stride(primals_151, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_152, (320, ), (1, ))
    assert_size_stride(primals_153, (320, ), (1, ))
    assert_size_stride(primals_154, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_155, (1280, ), (1, ))
    assert_size_stride(primals_156, (1280, ), (1, ))
    assert_size_stride(primals_157, (1000, 1280), (1280, 1))
    assert_size_stride(primals_158, (1000, ), (1, ))
    assert_size_stride(primals_159, (32, ), (1, ))
    assert_size_stride(primals_160, (32, ), (1, ))
    assert_size_stride(primals_161, (), ())
    assert_size_stride(primals_162, (32, ), (1, ))
    assert_size_stride(primals_163, (32, ), (1, ))
    assert_size_stride(primals_164, (), ())
    assert_size_stride(primals_165, (16, ), (1, ))
    assert_size_stride(primals_166, (16, ), (1, ))
    assert_size_stride(primals_167, (), ())
    assert_size_stride(primals_168, (48, ), (1, ))
    assert_size_stride(primals_169, (48, ), (1, ))
    assert_size_stride(primals_170, (), ())
    assert_size_stride(primals_171, (48, ), (1, ))
    assert_size_stride(primals_172, (48, ), (1, ))
    assert_size_stride(primals_173, (), ())
    assert_size_stride(primals_174, (24, ), (1, ))
    assert_size_stride(primals_175, (24, ), (1, ))
    assert_size_stride(primals_176, (), ())
    assert_size_stride(primals_177, (72, ), (1, ))
    assert_size_stride(primals_178, (72, ), (1, ))
    assert_size_stride(primals_179, (), ())
    assert_size_stride(primals_180, (72, ), (1, ))
    assert_size_stride(primals_181, (72, ), (1, ))
    assert_size_stride(primals_182, (), ())
    assert_size_stride(primals_183, (24, ), (1, ))
    assert_size_stride(primals_184, (24, ), (1, ))
    assert_size_stride(primals_185, (), ())
    assert_size_stride(primals_186, (72, ), (1, ))
    assert_size_stride(primals_187, (72, ), (1, ))
    assert_size_stride(primals_188, (), ())
    assert_size_stride(primals_189, (72, ), (1, ))
    assert_size_stride(primals_190, (72, ), (1, ))
    assert_size_stride(primals_191, (), ())
    assert_size_stride(primals_192, (24, ), (1, ))
    assert_size_stride(primals_193, (24, ), (1, ))
    assert_size_stride(primals_194, (), ())
    assert_size_stride(primals_195, (72, ), (1, ))
    assert_size_stride(primals_196, (72, ), (1, ))
    assert_size_stride(primals_197, (), ())
    assert_size_stride(primals_198, (72, ), (1, ))
    assert_size_stride(primals_199, (72, ), (1, ))
    assert_size_stride(primals_200, (), ())
    assert_size_stride(primals_201, (40, ), (1, ))
    assert_size_stride(primals_202, (40, ), (1, ))
    assert_size_stride(primals_203, (), ())
    assert_size_stride(primals_204, (120, ), (1, ))
    assert_size_stride(primals_205, (120, ), (1, ))
    assert_size_stride(primals_206, (), ())
    assert_size_stride(primals_207, (120, ), (1, ))
    assert_size_stride(primals_208, (120, ), (1, ))
    assert_size_stride(primals_209, (), ())
    assert_size_stride(primals_210, (40, ), (1, ))
    assert_size_stride(primals_211, (40, ), (1, ))
    assert_size_stride(primals_212, (), ())
    assert_size_stride(primals_213, (120, ), (1, ))
    assert_size_stride(primals_214, (120, ), (1, ))
    assert_size_stride(primals_215, (), ())
    assert_size_stride(primals_216, (120, ), (1, ))
    assert_size_stride(primals_217, (120, ), (1, ))
    assert_size_stride(primals_218, (), ())
    assert_size_stride(primals_219, (40, ), (1, ))
    assert_size_stride(primals_220, (40, ), (1, ))
    assert_size_stride(primals_221, (), ())
    assert_size_stride(primals_222, (240, ), (1, ))
    assert_size_stride(primals_223, (240, ), (1, ))
    assert_size_stride(primals_224, (), ())
    assert_size_stride(primals_225, (240, ), (1, ))
    assert_size_stride(primals_226, (240, ), (1, ))
    assert_size_stride(primals_227, (), ())
    assert_size_stride(primals_228, (80, ), (1, ))
    assert_size_stride(primals_229, (80, ), (1, ))
    assert_size_stride(primals_230, (), ())
    assert_size_stride(primals_231, (480, ), (1, ))
    assert_size_stride(primals_232, (480, ), (1, ))
    assert_size_stride(primals_233, (), ())
    assert_size_stride(primals_234, (480, ), (1, ))
    assert_size_stride(primals_235, (480, ), (1, ))
    assert_size_stride(primals_236, (), ())
    assert_size_stride(primals_237, (80, ), (1, ))
    assert_size_stride(primals_238, (80, ), (1, ))
    assert_size_stride(primals_239, (), ())
    assert_size_stride(primals_240, (480, ), (1, ))
    assert_size_stride(primals_241, (480, ), (1, ))
    assert_size_stride(primals_242, (), ())
    assert_size_stride(primals_243, (480, ), (1, ))
    assert_size_stride(primals_244, (480, ), (1, ))
    assert_size_stride(primals_245, (), ())
    assert_size_stride(primals_246, (80, ), (1, ))
    assert_size_stride(primals_247, (80, ), (1, ))
    assert_size_stride(primals_248, (), ())
    assert_size_stride(primals_249, (480, ), (1, ))
    assert_size_stride(primals_250, (480, ), (1, ))
    assert_size_stride(primals_251, (), ())
    assert_size_stride(primals_252, (480, ), (1, ))
    assert_size_stride(primals_253, (480, ), (1, ))
    assert_size_stride(primals_254, (), ())
    assert_size_stride(primals_255, (96, ), (1, ))
    assert_size_stride(primals_256, (96, ), (1, ))
    assert_size_stride(primals_257, (), ())
    assert_size_stride(primals_258, (576, ), (1, ))
    assert_size_stride(primals_259, (576, ), (1, ))
    assert_size_stride(primals_260, (), ())
    assert_size_stride(primals_261, (576, ), (1, ))
    assert_size_stride(primals_262, (576, ), (1, ))
    assert_size_stride(primals_263, (), ())
    assert_size_stride(primals_264, (96, ), (1, ))
    assert_size_stride(primals_265, (96, ), (1, ))
    assert_size_stride(primals_266, (), ())
    assert_size_stride(primals_267, (576, ), (1, ))
    assert_size_stride(primals_268, (576, ), (1, ))
    assert_size_stride(primals_269, (), ())
    assert_size_stride(primals_270, (576, ), (1, ))
    assert_size_stride(primals_271, (576, ), (1, ))
    assert_size_stride(primals_272, (), ())
    assert_size_stride(primals_273, (192, ), (1, ))
    assert_size_stride(primals_274, (192, ), (1, ))
    assert_size_stride(primals_275, (), ())
    assert_size_stride(primals_276, (1152, ), (1, ))
    assert_size_stride(primals_277, (1152, ), (1, ))
    assert_size_stride(primals_278, (), ())
    assert_size_stride(primals_279, (1152, ), (1, ))
    assert_size_stride(primals_280, (1152, ), (1, ))
    assert_size_stride(primals_281, (), ())
    assert_size_stride(primals_282, (192, ), (1, ))
    assert_size_stride(primals_283, (192, ), (1, ))
    assert_size_stride(primals_284, (), ())
    assert_size_stride(primals_285, (1152, ), (1, ))
    assert_size_stride(primals_286, (1152, ), (1, ))
    assert_size_stride(primals_287, (), ())
    assert_size_stride(primals_288, (1152, ), (1, ))
    assert_size_stride(primals_289, (1152, ), (1, ))
    assert_size_stride(primals_290, (), ())
    assert_size_stride(primals_291, (192, ), (1, ))
    assert_size_stride(primals_292, (192, ), (1, ))
    assert_size_stride(primals_293, (), ())
    assert_size_stride(primals_294, (1152, ), (1, ))
    assert_size_stride(primals_295, (1152, ), (1, ))
    assert_size_stride(primals_296, (), ())
    assert_size_stride(primals_297, (1152, ), (1, ))
    assert_size_stride(primals_298, (1152, ), (1, ))
    assert_size_stride(primals_299, (), ())
    assert_size_stride(primals_300, (192, ), (1, ))
    assert_size_stride(primals_301, (192, ), (1, ))
    assert_size_stride(primals_302, (), ())
    assert_size_stride(primals_303, (1152, ), (1, ))
    assert_size_stride(primals_304, (1152, ), (1, ))
    assert_size_stride(primals_305, (), ())
    assert_size_stride(primals_306, (1152, ), (1, ))
    assert_size_stride(primals_307, (1152, ), (1, ))
    assert_size_stride(primals_308, (), ())
    assert_size_stride(primals_309, (320, ), (1, ))
    assert_size_stride(primals_310, (320, ), (1, ))
    assert_size_stride(primals_311, (), ())
    assert_size_stride(primals_312, (1280, ), (1, ))
    assert_size_stride(primals_313, (1280, ), (1, ))
    assert_size_stride(primals_314, (), ())
    assert_size_stride(primals_315, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_1
    del primals_315
    # Source Nodes: [l__mod___layers_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (4, 32, 112, 112), (401408, 1, 3584, 32))
    buf3 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_1(c_void_p(buf2.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf3.data_ptr()))
    del primals_3
    # Source Nodes: [l__mod___layers_3], Original ATen: [aten.convolution]
    buf4 = extern_kernels.convolution(buf3, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf4, (4, 32, 112, 112), (401408, 1, 3584, 32))
    buf5 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_2(c_void_p(buf4.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf5.data_ptr()))
    del primals_6
    # Source Nodes: [l__mod___layers_6], Original ATen: [aten.convolution]
    buf6 = extern_kernels.convolution(buf5, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf6, (4, 16, 112, 112), (200704, 1, 1792, 16))
    buf7 = empty_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_3(c_void_p(buf6.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf7.data_ptr()))
    del primals_9
    # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_0], Original ATen: [aten.convolution]
    buf8 = extern_kernels.convolution(buf7, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (4, 48, 112, 112), (602112, 1, 5376, 48))
    buf9 = empty_strided((4, 48, 112, 112), (602112, 1, 5376, 48), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_4(c_void_p(buf8.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf9.data_ptr()))
    del primals_12
    # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_3], Original ATen: [aten.convolution]
    buf10 = extern_kernels.convolution(buf9, primals_13, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
    assert_size_stride(buf10, (4, 48, 56, 56), (150528, 1, 2688, 48))
    buf11 = empty_strided((4, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_5(c_void_p(buf10.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf11.data_ptr()))
    del primals_15
    # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_6], Original ATen: [aten.convolution]
    buf12 = extern_kernels.convolution(buf11, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf12, (4, 24, 56, 56), (75264, 1, 1344, 24))
    buf13 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_6(c_void_p(buf12.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf13.data_ptr()))
    del primals_18
    # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_0], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf13, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf14, (4, 72, 56, 56), (225792, 1, 4032, 72))
    buf15 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_7(c_void_p(buf14.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf15.data_ptr()))
    del primals_21
    # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_3], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf15, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
    assert_size_stride(buf16, (4, 72, 56, 56), (225792, 1, 4032, 72))
    buf17 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_8(c_void_p(buf16.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf17.data_ptr()))
    del primals_24
    # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_6], Original ATen: [aten.convolution]
    buf18 = extern_kernels.convolution(buf17, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf18, (4, 24, 56, 56), (75264, 1, 1344, 24))
    buf19 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_9(c_void_p(buf18.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf19.data_ptr()))
    del primals_27
    # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_0], Original ATen: [aten.convolution]
    buf20 = extern_kernels.convolution(buf19, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf20, (4, 72, 56, 56), (225792, 1, 4032, 72))
    buf21 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_10(c_void_p(buf20.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf21.data_ptr()))
    del primals_30
    # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_3], Original ATen: [aten.convolution]
    buf22 = extern_kernels.convolution(buf21, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
    assert_size_stride(buf22, (4, 72, 56, 56), (225792, 1, 4032, 72))
    buf23 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_11(c_void_p(buf22.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf23.data_ptr()))
    del primals_33
    # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_6], Original ATen: [aten.convolution]
    buf24 = extern_kernels.convolution(buf23, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf24, (4, 24, 56, 56), (75264, 1, 1344, 24))
    buf25 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_12(c_void_p(buf24.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf25.data_ptr()))
    del primals_36
    # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_0], Original ATen: [aten.convolution]
    buf26 = extern_kernels.convolution(buf25, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf26, (4, 72, 56, 56), (225792, 1, 4032, 72))
    buf27 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_13(c_void_p(buf26.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf27.data_ptr()))
    del primals_39
    # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_3], Original ATen: [aten.convolution]
    buf28 = extern_kernels.convolution(buf27, primals_40, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
    assert_size_stride(buf28, (4, 72, 28, 28), (56448, 1, 2016, 72))
    buf29 = empty_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_14(c_void_p(buf28.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf29.data_ptr()))
    del primals_42
    # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_6], Original ATen: [aten.convolution]
    buf30 = extern_kernels.convolution(buf29, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf30, (4, 40, 28, 28), (31360, 1, 1120, 40))
    buf31 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_15(c_void_p(buf30.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf31.data_ptr()))
    del primals_45
    # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_0], Original ATen: [aten.convolution]
    buf32 = extern_kernels.convolution(buf31, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf32, (4, 120, 28, 28), (94080, 1, 3360, 120))
    buf33 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_16(c_void_p(buf32.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf33.data_ptr()))
    del primals_48
    # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_3], Original ATen: [aten.convolution]
    buf34 = extern_kernels.convolution(buf33, primals_49, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf34, (4, 120, 28, 28), (94080, 1, 3360, 120))
    buf35 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_17(c_void_p(buf34.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf35.data_ptr()))
    del primals_51
    # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_6], Original ATen: [aten.convolution]
    buf36 = extern_kernels.convolution(buf35, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf36, (4, 40, 28, 28), (31360, 1, 1120, 40))
    buf37 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_18(c_void_p(buf36.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf37.data_ptr()))
    del primals_54
    # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_0], Original ATen: [aten.convolution]
    buf38 = extern_kernels.convolution(buf37, primals_55, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf38, (4, 120, 28, 28), (94080, 1, 3360, 120))
    buf39 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_19(c_void_p(buf38.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf39.data_ptr()))
    del primals_57
    # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_3], Original ATen: [aten.convolution]
    buf40 = extern_kernels.convolution(buf39, primals_58, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf40, (4, 120, 28, 28), (94080, 1, 3360, 120))
    buf41 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_20(c_void_p(buf40.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf41.data_ptr()))
    del primals_60
    # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_6], Original ATen: [aten.convolution]
    buf42 = extern_kernels.convolution(buf41, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf42, (4, 40, 28, 28), (31360, 1, 1120, 40))
    buf43 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_21(c_void_p(buf42.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf43.data_ptr()))
    del primals_63
    # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_0], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf43, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf44, (4, 240, 28, 28), (188160, 1, 6720, 240))
    buf45 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_22(c_void_p(buf44.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf45.data_ptr()))
    del primals_66
    # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_3], Original ATen: [aten.convolution]
    buf46 = extern_kernels.convolution(buf45, primals_67, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf46, (4, 240, 14, 14), (47040, 1, 3360, 240))
    buf47 = empty_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_23(c_void_p(buf46.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf47.data_ptr()))
    del primals_69
    # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_6], Original ATen: [aten.convolution]
    buf48 = extern_kernels.convolution(buf47, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf48, (4, 80, 14, 14), (15680, 1, 1120, 80))
    buf49 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_24(c_void_p(buf48.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf49.data_ptr()))
    del primals_72
    # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_0], Original ATen: [aten.convolution]
    buf50 = extern_kernels.convolution(buf49, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf50, (4, 480, 14, 14), (94080, 1, 6720, 480))
    buf51 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_25(c_void_p(buf50.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf51.data_ptr()))
    del primals_75
    # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_3], Original ATen: [aten.convolution]
    buf52 = extern_kernels.convolution(buf51, primals_76, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf52, (4, 480, 14, 14), (94080, 1, 6720, 480))
    buf53 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_26(c_void_p(buf52.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf53.data_ptr()))
    del primals_78
    # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_6], Original ATen: [aten.convolution]
    buf54 = extern_kernels.convolution(buf53, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf54, (4, 80, 14, 14), (15680, 1, 1120, 80))
    buf55 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_27(c_void_p(buf54.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf55.data_ptr()))
    del primals_81
    # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_0], Original ATen: [aten.convolution]
    buf56 = extern_kernels.convolution(buf55, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf56, (4, 480, 14, 14), (94080, 1, 6720, 480))
    buf57 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_28(c_void_p(buf56.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf57.data_ptr()))
    del primals_84
    # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_3], Original ATen: [aten.convolution]
    buf58 = extern_kernels.convolution(buf57, primals_85, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf58, (4, 480, 14, 14), (94080, 1, 6720, 480))
    buf59 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_29(c_void_p(buf58.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf59.data_ptr()))
    del primals_87
    # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_6], Original ATen: [aten.convolution]
    buf60 = extern_kernels.convolution(buf59, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf60, (4, 80, 14, 14), (15680, 1, 1120, 80))
    buf61 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_30(c_void_p(buf60.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf61.data_ptr()))
    del primals_90
    # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_0], Original ATen: [aten.convolution]
    buf62 = extern_kernels.convolution(buf61, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf62, (4, 480, 14, 14), (94080, 1, 6720, 480))
    buf63 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_31(c_void_p(buf62.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf63.data_ptr()))
    del primals_93
    # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_3], Original ATen: [aten.convolution]
    buf64 = extern_kernels.convolution(buf63, primals_94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf64, (4, 480, 14, 14), (94080, 1, 6720, 480))
    buf65 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_32(c_void_p(buf64.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf65.data_ptr()))
    del primals_96
    # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_6], Original ATen: [aten.convolution]
    buf66 = extern_kernels.convolution(buf65, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf66, (4, 96, 14, 14), (18816, 1, 1344, 96))
    buf67 = empty_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_33(c_void_p(buf66.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf67.data_ptr()))
    del primals_99
    # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_0], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (4, 576, 14, 14), (112896, 1, 8064, 576))
    buf69 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_34(c_void_p(buf68.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf69.data_ptr()))
    del primals_102
    # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_3], Original ATen: [aten.convolution]
    buf70 = extern_kernels.convolution(buf69, primals_103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
    assert_size_stride(buf70, (4, 576, 14, 14), (112896, 1, 8064, 576))
    buf71 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_35(c_void_p(buf70.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf71.data_ptr()))
    del primals_105
    # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_6], Original ATen: [aten.convolution]
    buf72 = extern_kernels.convolution(buf71, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf72, (4, 96, 14, 14), (18816, 1, 1344, 96))
    buf73 = empty_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_36(c_void_p(buf72.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf73.data_ptr()))
    del primals_108
    # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_0], Original ATen: [aten.convolution]
    buf74 = extern_kernels.convolution(buf73, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf74, (4, 576, 14, 14), (112896, 1, 8064, 576))
    buf75 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_37(c_void_p(buf74.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf75.data_ptr()))
    del primals_111
    # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_3], Original ATen: [aten.convolution]
    buf76 = extern_kernels.convolution(buf75, primals_112, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
    assert_size_stride(buf76, (4, 576, 7, 7), (28224, 1, 4032, 576))
    buf77 = empty_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_38(c_void_p(buf76.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf77.data_ptr()))
    del primals_114
    # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_6], Original ATen: [aten.convolution]
    buf78 = extern_kernels.convolution(buf77, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf78, (4, 192, 7, 7), (9408, 1, 1344, 192))
    buf79 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_39(c_void_p(buf78.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(buf79.data_ptr()))
    del primals_117
    # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_0], Original ATen: [aten.convolution]
    buf80 = extern_kernels.convolution(buf79, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf80, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    buf81 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_40(c_void_p(buf80.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(buf81.data_ptr()))
    del primals_120
    # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_3], Original ATen: [aten.convolution]
    buf82 = extern_kernels.convolution(buf81, primals_121, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
    assert_size_stride(buf82, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    buf83 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_41(c_void_p(buf82.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(buf83.data_ptr()))
    del primals_123
    # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_6], Original ATen: [aten.convolution]
    buf84 = extern_kernels.convolution(buf83, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (4, 192, 7, 7), (9408, 1, 1344, 192))
    buf85 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_42(c_void_p(buf84.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf85.data_ptr()))
    del primals_126
    # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_0], Original ATen: [aten.convolution]
    buf86 = extern_kernels.convolution(buf85, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf86, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    buf87 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_43(c_void_p(buf86.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(buf87.data_ptr()))
    del primals_129
    # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_3], Original ATen: [aten.convolution]
    buf88 = extern_kernels.convolution(buf87, primals_130, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
    assert_size_stride(buf88, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    buf89 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_44(c_void_p(buf88.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(buf89.data_ptr()))
    del primals_132
    # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_6], Original ATen: [aten.convolution]
    buf90 = extern_kernels.convolution(buf89, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf90, (4, 192, 7, 7), (9408, 1, 1344, 192))
    buf91 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_45(c_void_p(buf90.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf91.data_ptr()))
    del primals_135
    # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_0], Original ATen: [aten.convolution]
    buf92 = extern_kernels.convolution(buf91, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf92, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    buf93 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_46(c_void_p(buf92.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(buf93.data_ptr()))
    del primals_138
    # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_3], Original ATen: [aten.convolution]
    buf94 = extern_kernels.convolution(buf93, primals_139, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
    assert_size_stride(buf94, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    buf95 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_47(c_void_p(buf94.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(buf95.data_ptr()))
    del primals_141
    # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_6], Original ATen: [aten.convolution]
    buf96 = extern_kernels.convolution(buf95, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf96, (4, 192, 7, 7), (9408, 1, 1344, 192))
    buf97 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_48(c_void_p(buf96.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf97.data_ptr()))
    del primals_144
    # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_0], Original ATen: [aten.convolution]
    buf98 = extern_kernels.convolution(buf97, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf98, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    buf99 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_49(c_void_p(buf98.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf99.data_ptr()))
    del primals_147
    # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_3], Original ATen: [aten.convolution]
    buf100 = extern_kernels.convolution(buf99, primals_148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
    assert_size_stride(buf100, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    buf101 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_50(c_void_p(buf100.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(buf101.data_ptr()))
    del primals_150
    # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_6], Original ATen: [aten.convolution]
    buf102 = extern_kernels.convolution(buf101, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (4, 320, 7, 7), (15680, 1, 2240, 320))
    buf103 = empty_strided((4, 320, 7, 7), (15680, 1, 2240, 320), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_51(c_void_p(buf102.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf103.data_ptr()))
    del primals_153
    # Source Nodes: [l__mod___layers_14], Original ATen: [aten.convolution]
    buf104 = extern_kernels.convolution(buf103, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf104, (4, 1280, 7, 7), (62720, 1, 8960, 1280))
    buf105 = empty_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.float32)
    buf106 = empty((4, 1280), device='cpu', dtype=torch.float32)
    buf107 = buf106; del buf106  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_52(c_void_p(buf107.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(buf105.data_ptr()))
    del primals_156
    buf108 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_158, buf107, reinterpret_tensor(primals_157, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf108)
    del primals_158
    buf109 = empty_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.bool)
    cpp_fused_threshold_backward_53(c_void_p(buf105.data_ptr()), c_void_p(buf109.data_ptr()))
    return (buf108, buf0, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf107, reinterpret_tensor(primals_157, (1000, 1280), (1280, 1), 0), buf109, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((48, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((96, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((576, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((192, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_162 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_165 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_168 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_171 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_174 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_177 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_180 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_183 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_186 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_189 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_192 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_195 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_198 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_201 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_204 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_207 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_210 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_213 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_216 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_219 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_222 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_225 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_228 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_231 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_234 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_237 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_240 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_243 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_246 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_249 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_252 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_255 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_258 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_261 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_264 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_267 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_270 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_273 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_276 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_279 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_282 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_285 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_288 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_291 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_294 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_297 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_300 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_303 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_306 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_309 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_312 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_315 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mnasnet1_0', benchmark_compiled_module)
