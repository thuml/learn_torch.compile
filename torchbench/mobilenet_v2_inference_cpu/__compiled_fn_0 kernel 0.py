
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


cpp_fused__native_batch_norm_legit_no_training_hardtanh_1 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_2 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_3 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_hardtanh_4 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_5 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_hardtanh_7 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_8 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_10 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_11 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_13 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_14 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_16 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_17 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_18 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_19 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_20 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_21 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_22 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_23 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_25 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_26 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_27 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_28 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_29 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_31 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_32 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_33 = async_compile.cpp('''
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
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_34 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_35 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_37 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_38 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_39 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_40 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_41 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
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
                auto tmp17 = static_cast<float>(0.0);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = at::vec::maximum(tmp16, tmp18);
                auto tmp20 = static_cast<float>(6.0);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = at::vec::minimum(tmp19, tmp21);
                tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_42 = async_compile.cpp('''
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_43 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_44 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_46 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_47 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_48 = async_compile.cpp('''
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
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_49 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardtanh_50 = async_compile.cpp('''
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
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = at::vec::maximum(tmp16, tmp18);
                    auto tmp20 = static_cast<float>(6.0);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = at::vec::minimum(tmp19, tmp21);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_hardtanh_mean_52 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1280L*x2) + (62720L*x0)));
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
                            auto tmp17 = static_cast<float>(0.0);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = at::vec::maximum(tmp16, tmp18);
                            auto tmp20 = static_cast<float>(6.0);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = at::vec::minimum(tmp19, tmp21);
                            tmp_acc0_vec = tmp_acc0_vec + tmp22;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1280L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5120L); x0+=static_cast<long>(8L))
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (16, ), (1, ))
    assert_size_stride(arg9_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg10_1, (96, ), (1, ))
    assert_size_stride(arg11_1, (96, ), (1, ))
    assert_size_stride(arg12_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg13_1, (96, ), (1, ))
    assert_size_stride(arg14_1, (96, ), (1, ))
    assert_size_stride(arg15_1, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg16_1, (24, ), (1, ))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg19_1, (144, ), (1, ))
    assert_size_stride(arg20_1, (144, ), (1, ))
    assert_size_stride(arg21_1, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg22_1, (144, ), (1, ))
    assert_size_stride(arg23_1, (144, ), (1, ))
    assert_size_stride(arg24_1, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg25_1, (24, ), (1, ))
    assert_size_stride(arg26_1, (24, ), (1, ))
    assert_size_stride(arg27_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg28_1, (144, ), (1, ))
    assert_size_stride(arg29_1, (144, ), (1, ))
    assert_size_stride(arg30_1, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg31_1, (144, ), (1, ))
    assert_size_stride(arg32_1, (144, ), (1, ))
    assert_size_stride(arg33_1, (32, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg34_1, (32, ), (1, ))
    assert_size_stride(arg35_1, (32, ), (1, ))
    assert_size_stride(arg36_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg37_1, (192, ), (1, ))
    assert_size_stride(arg38_1, (192, ), (1, ))
    assert_size_stride(arg39_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg40_1, (192, ), (1, ))
    assert_size_stride(arg41_1, (192, ), (1, ))
    assert_size_stride(arg42_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg43_1, (32, ), (1, ))
    assert_size_stride(arg44_1, (32, ), (1, ))
    assert_size_stride(arg45_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg46_1, (192, ), (1, ))
    assert_size_stride(arg47_1, (192, ), (1, ))
    assert_size_stride(arg48_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg49_1, (192, ), (1, ))
    assert_size_stride(arg50_1, (192, ), (1, ))
    assert_size_stride(arg51_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg52_1, (32, ), (1, ))
    assert_size_stride(arg53_1, (32, ), (1, ))
    assert_size_stride(arg54_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg55_1, (192, ), (1, ))
    assert_size_stride(arg56_1, (192, ), (1, ))
    assert_size_stride(arg57_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg58_1, (192, ), (1, ))
    assert_size_stride(arg59_1, (192, ), (1, ))
    assert_size_stride(arg60_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg61_1, (64, ), (1, ))
    assert_size_stride(arg62_1, (64, ), (1, ))
    assert_size_stride(arg63_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg64_1, (384, ), (1, ))
    assert_size_stride(arg65_1, (384, ), (1, ))
    assert_size_stride(arg66_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg67_1, (384, ), (1, ))
    assert_size_stride(arg68_1, (384, ), (1, ))
    assert_size_stride(arg69_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg70_1, (64, ), (1, ))
    assert_size_stride(arg71_1, (64, ), (1, ))
    assert_size_stride(arg72_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg73_1, (384, ), (1, ))
    assert_size_stride(arg74_1, (384, ), (1, ))
    assert_size_stride(arg75_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg76_1, (384, ), (1, ))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg79_1, (64, ), (1, ))
    assert_size_stride(arg80_1, (64, ), (1, ))
    assert_size_stride(arg81_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg82_1, (384, ), (1, ))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg85_1, (384, ), (1, ))
    assert_size_stride(arg86_1, (384, ), (1, ))
    assert_size_stride(arg87_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg88_1, (64, ), (1, ))
    assert_size_stride(arg89_1, (64, ), (1, ))
    assert_size_stride(arg90_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg91_1, (384, ), (1, ))
    assert_size_stride(arg92_1, (384, ), (1, ))
    assert_size_stride(arg93_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg97_1, (96, ), (1, ))
    assert_size_stride(arg98_1, (96, ), (1, ))
    assert_size_stride(arg99_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg100_1, (576, ), (1, ))
    assert_size_stride(arg101_1, (576, ), (1, ))
    assert_size_stride(arg102_1, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg103_1, (576, ), (1, ))
    assert_size_stride(arg104_1, (576, ), (1, ))
    assert_size_stride(arg105_1, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg106_1, (96, ), (1, ))
    assert_size_stride(arg107_1, (96, ), (1, ))
    assert_size_stride(arg108_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg109_1, (576, ), (1, ))
    assert_size_stride(arg110_1, (576, ), (1, ))
    assert_size_stride(arg111_1, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg112_1, (576, ), (1, ))
    assert_size_stride(arg113_1, (576, ), (1, ))
    assert_size_stride(arg114_1, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg115_1, (96, ), (1, ))
    assert_size_stride(arg116_1, (96, ), (1, ))
    assert_size_stride(arg117_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg118_1, (576, ), (1, ))
    assert_size_stride(arg119_1, (576, ), (1, ))
    assert_size_stride(arg120_1, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg121_1, (576, ), (1, ))
    assert_size_stride(arg122_1, (576, ), (1, ))
    assert_size_stride(arg123_1, (160, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg124_1, (160, ), (1, ))
    assert_size_stride(arg125_1, (160, ), (1, ))
    assert_size_stride(arg126_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg127_1, (960, ), (1, ))
    assert_size_stride(arg128_1, (960, ), (1, ))
    assert_size_stride(arg129_1, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg130_1, (960, ), (1, ))
    assert_size_stride(arg131_1, (960, ), (1, ))
    assert_size_stride(arg132_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg133_1, (160, ), (1, ))
    assert_size_stride(arg134_1, (160, ), (1, ))
    assert_size_stride(arg135_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg136_1, (960, ), (1, ))
    assert_size_stride(arg137_1, (960, ), (1, ))
    assert_size_stride(arg138_1, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg139_1, (960, ), (1, ))
    assert_size_stride(arg140_1, (960, ), (1, ))
    assert_size_stride(arg141_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg142_1, (160, ), (1, ))
    assert_size_stride(arg143_1, (160, ), (1, ))
    assert_size_stride(arg144_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg145_1, (960, ), (1, ))
    assert_size_stride(arg146_1, (960, ), (1, ))
    assert_size_stride(arg147_1, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg148_1, (960, ), (1, ))
    assert_size_stride(arg149_1, (960, ), (1, ))
    assert_size_stride(arg150_1, (320, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg151_1, (320, ), (1, ))
    assert_size_stride(arg152_1, (320, ), (1, ))
    assert_size_stride(arg153_1, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg154_1, (1280, ), (1, ))
    assert_size_stride(arg155_1, (1280, ), (1, ))
    assert_size_stride(arg156_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg157_1, (1000, ), (1, ))
    assert_size_stride(arg158_1, (32, ), (1, ))
    assert_size_stride(arg159_1, (32, ), (1, ))
    assert_size_stride(arg160_1, (), ())
    assert_size_stride(arg161_1, (32, ), (1, ))
    assert_size_stride(arg162_1, (32, ), (1, ))
    assert_size_stride(arg163_1, (), ())
    assert_size_stride(arg164_1, (16, ), (1, ))
    assert_size_stride(arg165_1, (16, ), (1, ))
    assert_size_stride(arg166_1, (), ())
    assert_size_stride(arg167_1, (96, ), (1, ))
    assert_size_stride(arg168_1, (96, ), (1, ))
    assert_size_stride(arg169_1, (), ())
    assert_size_stride(arg170_1, (96, ), (1, ))
    assert_size_stride(arg171_1, (96, ), (1, ))
    assert_size_stride(arg172_1, (), ())
    assert_size_stride(arg173_1, (24, ), (1, ))
    assert_size_stride(arg174_1, (24, ), (1, ))
    assert_size_stride(arg175_1, (), ())
    assert_size_stride(arg176_1, (144, ), (1, ))
    assert_size_stride(arg177_1, (144, ), (1, ))
    assert_size_stride(arg178_1, (), ())
    assert_size_stride(arg179_1, (144, ), (1, ))
    assert_size_stride(arg180_1, (144, ), (1, ))
    assert_size_stride(arg181_1, (), ())
    assert_size_stride(arg182_1, (24, ), (1, ))
    assert_size_stride(arg183_1, (24, ), (1, ))
    assert_size_stride(arg184_1, (), ())
    assert_size_stride(arg185_1, (144, ), (1, ))
    assert_size_stride(arg186_1, (144, ), (1, ))
    assert_size_stride(arg187_1, (), ())
    assert_size_stride(arg188_1, (144, ), (1, ))
    assert_size_stride(arg189_1, (144, ), (1, ))
    assert_size_stride(arg190_1, (), ())
    assert_size_stride(arg191_1, (32, ), (1, ))
    assert_size_stride(arg192_1, (32, ), (1, ))
    assert_size_stride(arg193_1, (), ())
    assert_size_stride(arg194_1, (192, ), (1, ))
    assert_size_stride(arg195_1, (192, ), (1, ))
    assert_size_stride(arg196_1, (), ())
    assert_size_stride(arg197_1, (192, ), (1, ))
    assert_size_stride(arg198_1, (192, ), (1, ))
    assert_size_stride(arg199_1, (), ())
    assert_size_stride(arg200_1, (32, ), (1, ))
    assert_size_stride(arg201_1, (32, ), (1, ))
    assert_size_stride(arg202_1, (), ())
    assert_size_stride(arg203_1, (192, ), (1, ))
    assert_size_stride(arg204_1, (192, ), (1, ))
    assert_size_stride(arg205_1, (), ())
    assert_size_stride(arg206_1, (192, ), (1, ))
    assert_size_stride(arg207_1, (192, ), (1, ))
    assert_size_stride(arg208_1, (), ())
    assert_size_stride(arg209_1, (32, ), (1, ))
    assert_size_stride(arg210_1, (32, ), (1, ))
    assert_size_stride(arg211_1, (), ())
    assert_size_stride(arg212_1, (192, ), (1, ))
    assert_size_stride(arg213_1, (192, ), (1, ))
    assert_size_stride(arg214_1, (), ())
    assert_size_stride(arg215_1, (192, ), (1, ))
    assert_size_stride(arg216_1, (192, ), (1, ))
    assert_size_stride(arg217_1, (), ())
    assert_size_stride(arg218_1, (64, ), (1, ))
    assert_size_stride(arg219_1, (64, ), (1, ))
    assert_size_stride(arg220_1, (), ())
    assert_size_stride(arg221_1, (384, ), (1, ))
    assert_size_stride(arg222_1, (384, ), (1, ))
    assert_size_stride(arg223_1, (), ())
    assert_size_stride(arg224_1, (384, ), (1, ))
    assert_size_stride(arg225_1, (384, ), (1, ))
    assert_size_stride(arg226_1, (), ())
    assert_size_stride(arg227_1, (64, ), (1, ))
    assert_size_stride(arg228_1, (64, ), (1, ))
    assert_size_stride(arg229_1, (), ())
    assert_size_stride(arg230_1, (384, ), (1, ))
    assert_size_stride(arg231_1, (384, ), (1, ))
    assert_size_stride(arg232_1, (), ())
    assert_size_stride(arg233_1, (384, ), (1, ))
    assert_size_stride(arg234_1, (384, ), (1, ))
    assert_size_stride(arg235_1, (), ())
    assert_size_stride(arg236_1, (64, ), (1, ))
    assert_size_stride(arg237_1, (64, ), (1, ))
    assert_size_stride(arg238_1, (), ())
    assert_size_stride(arg239_1, (384, ), (1, ))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (), ())
    assert_size_stride(arg242_1, (384, ), (1, ))
    assert_size_stride(arg243_1, (384, ), (1, ))
    assert_size_stride(arg244_1, (), ())
    assert_size_stride(arg245_1, (64, ), (1, ))
    assert_size_stride(arg246_1, (64, ), (1, ))
    assert_size_stride(arg247_1, (), ())
    assert_size_stride(arg248_1, (384, ), (1, ))
    assert_size_stride(arg249_1, (384, ), (1, ))
    assert_size_stride(arg250_1, (), ())
    assert_size_stride(arg251_1, (384, ), (1, ))
    assert_size_stride(arg252_1, (384, ), (1, ))
    assert_size_stride(arg253_1, (), ())
    assert_size_stride(arg254_1, (96, ), (1, ))
    assert_size_stride(arg255_1, (96, ), (1, ))
    assert_size_stride(arg256_1, (), ())
    assert_size_stride(arg257_1, (576, ), (1, ))
    assert_size_stride(arg258_1, (576, ), (1, ))
    assert_size_stride(arg259_1, (), ())
    assert_size_stride(arg260_1, (576, ), (1, ))
    assert_size_stride(arg261_1, (576, ), (1, ))
    assert_size_stride(arg262_1, (), ())
    assert_size_stride(arg263_1, (96, ), (1, ))
    assert_size_stride(arg264_1, (96, ), (1, ))
    assert_size_stride(arg265_1, (), ())
    assert_size_stride(arg266_1, (576, ), (1, ))
    assert_size_stride(arg267_1, (576, ), (1, ))
    assert_size_stride(arg268_1, (), ())
    assert_size_stride(arg269_1, (576, ), (1, ))
    assert_size_stride(arg270_1, (576, ), (1, ))
    assert_size_stride(arg271_1, (), ())
    assert_size_stride(arg272_1, (96, ), (1, ))
    assert_size_stride(arg273_1, (96, ), (1, ))
    assert_size_stride(arg274_1, (), ())
    assert_size_stride(arg275_1, (576, ), (1, ))
    assert_size_stride(arg276_1, (576, ), (1, ))
    assert_size_stride(arg277_1, (), ())
    assert_size_stride(arg278_1, (576, ), (1, ))
    assert_size_stride(arg279_1, (576, ), (1, ))
    assert_size_stride(arg280_1, (), ())
    assert_size_stride(arg281_1, (160, ), (1, ))
    assert_size_stride(arg282_1, (160, ), (1, ))
    assert_size_stride(arg283_1, (), ())
    assert_size_stride(arg284_1, (960, ), (1, ))
    assert_size_stride(arg285_1, (960, ), (1, ))
    assert_size_stride(arg286_1, (), ())
    assert_size_stride(arg287_1, (960, ), (1, ))
    assert_size_stride(arg288_1, (960, ), (1, ))
    assert_size_stride(arg289_1, (), ())
    assert_size_stride(arg290_1, (160, ), (1, ))
    assert_size_stride(arg291_1, (160, ), (1, ))
    assert_size_stride(arg292_1, (), ())
    assert_size_stride(arg293_1, (960, ), (1, ))
    assert_size_stride(arg294_1, (960, ), (1, ))
    assert_size_stride(arg295_1, (), ())
    assert_size_stride(arg296_1, (960, ), (1, ))
    assert_size_stride(arg297_1, (960, ), (1, ))
    assert_size_stride(arg298_1, (), ())
    assert_size_stride(arg299_1, (160, ), (1, ))
    assert_size_stride(arg300_1, (160, ), (1, ))
    assert_size_stride(arg301_1, (), ())
    assert_size_stride(arg302_1, (960, ), (1, ))
    assert_size_stride(arg303_1, (960, ), (1, ))
    assert_size_stride(arg304_1, (), ())
    assert_size_stride(arg305_1, (960, ), (1, ))
    assert_size_stride(arg306_1, (960, ), (1, ))
    assert_size_stride(arg307_1, (), ())
    assert_size_stride(arg308_1, (320, ), (1, ))
    assert_size_stride(arg309_1, (320, ), (1, ))
    assert_size_stride(arg310_1, (), ())
    assert_size_stride(arg311_1, (1280, ), (1, ))
    assert_size_stride(arg312_1, (1280, ), (1, ))
    assert_size_stride(arg313_1, (), ())
    assert_size_stride(arg314_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg314_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg314_1
    # Source Nodes: [l__mod___features_0_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (4, 32, 112, 112), (401408, 1, 3584, 32))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_1(c_void_p(buf3.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()))
    del arg158_1
    del arg159_1
    del arg1_1
    del arg2_1
    # Source Nodes: [getattr_l__mod___features___1___conv_0_0, l__mod___features_0_1, l__mod___features_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf4 = extern_kernels.convolution(buf3, arg3_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf4, (4, 32, 112, 112), (401408, 1, 3584, 32))
    del arg3_1
    del buf3
    buf5 = buf4; del buf4  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_2(c_void_p(buf5.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg161_1
    del arg162_1
    del arg4_1
    del arg5_1
    # Source Nodes: [getattr_l__mod___features___1___conv_0_1, getattr_l__mod___features___1___conv_0_2, getattr_l__mod___features___1___conv_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf6 = extern_kernels.convolution(buf5, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf6, (4, 16, 112, 112), (200704, 1, 1792, 16))
    del arg6_1
    del buf5
    buf7 = buf6; del buf6  # reuse
    cpp_fused__native_batch_norm_legit_no_training_3(c_void_p(buf7.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()))
    del arg164_1
    del arg165_1
    del arg7_1
    del arg8_1
    # Source Nodes: [getattr_l__mod___features___1___conv_2, getattr_l__mod___features___2___conv_0_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf8 = extern_kernels.convolution(buf7, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (4, 96, 112, 112), (1204224, 1, 10752, 96))
    del arg9_1
    del buf7
    buf9 = buf8; del buf8  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_4(c_void_p(buf9.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()))
    del arg10_1
    del arg11_1
    del arg167_1
    del arg168_1
    # Source Nodes: [getattr_l__mod___features___2___conv_0_1, getattr_l__mod___features___2___conv_0_2, getattr_l__mod___features___2___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf10 = extern_kernels.convolution(buf9, arg12_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
    assert_size_stride(buf10, (4, 96, 56, 56), (301056, 1, 5376, 96))
    del arg12_1
    del buf9
    buf11 = buf10; del buf10  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_5(c_void_p(buf11.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()))
    del arg13_1
    del arg14_1
    del arg170_1
    del arg171_1
    # Source Nodes: [getattr_l__mod___features___2___conv_1_1, getattr_l__mod___features___2___conv_1_2, getattr_l__mod___features___2___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf12 = extern_kernels.convolution(buf11, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf12, (4, 24, 56, 56), (75264, 1, 1344, 24))
    del arg15_1
    del buf11
    buf13 = buf12; del buf12  # reuse
    cpp_fused__native_batch_norm_legit_no_training_6(c_void_p(buf13.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg16_1
    del arg173_1
    del arg174_1
    del arg17_1
    # Source Nodes: [getattr_l__mod___features___3___conv_0_0], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf13, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf14, (4, 144, 56, 56), (451584, 1, 8064, 144))
    del arg18_1
    buf15 = buf14; del buf14  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_7(c_void_p(buf15.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()))
    del arg176_1
    del arg177_1
    del arg19_1
    del arg20_1
    # Source Nodes: [getattr_l__mod___features___3___conv_0_1, getattr_l__mod___features___3___conv_0_2, getattr_l__mod___features___3___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf16 = extern_kernels.convolution(buf15, arg21_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
    assert_size_stride(buf16, (4, 144, 56, 56), (451584, 1, 8064, 144))
    del arg21_1
    del buf15
    buf17 = buf16; del buf16  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_8(c_void_p(buf17.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg179_1
    del arg180_1
    del arg22_1
    del arg23_1
    # Source Nodes: [getattr_l__mod___features___3___conv_1_1, getattr_l__mod___features___3___conv_1_2, getattr_l__mod___features___3___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf18 = extern_kernels.convolution(buf17, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf18, (4, 24, 56, 56), (75264, 1, 1344, 24))
    del arg24_1
    del buf17
    buf19 = buf13; del buf13  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_9(c_void_p(buf19.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()))
    del arg182_1
    del arg183_1
    del arg25_1
    del arg26_1
    del buf18
    # Source Nodes: [add, getattr_l__mod___features___3___conv_3, getattr_l__mod___features___4___conv_0_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf20 = extern_kernels.convolution(buf19, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf20, (4, 144, 56, 56), (451584, 1, 8064, 144))
    del arg27_1
    del buf19
    buf21 = buf20; del buf20  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_10(c_void_p(buf21.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg185_1
    del arg186_1
    del arg28_1
    del arg29_1
    # Source Nodes: [getattr_l__mod___features___4___conv_0_1, getattr_l__mod___features___4___conv_0_2, getattr_l__mod___features___4___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf22 = extern_kernels.convolution(buf21, arg30_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
    assert_size_stride(buf22, (4, 144, 28, 28), (112896, 1, 4032, 144))
    del arg30_1
    del buf21
    buf23 = buf22; del buf22  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_11(c_void_p(buf23.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()))
    del arg188_1
    del arg189_1
    del arg31_1
    del arg32_1
    # Source Nodes: [getattr_l__mod___features___4___conv_1_1, getattr_l__mod___features___4___conv_1_2, getattr_l__mod___features___4___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf24 = extern_kernels.convolution(buf23, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf24, (4, 32, 28, 28), (25088, 1, 896, 32))
    del arg33_1
    del buf23
    buf25 = buf24; del buf24  # reuse
    cpp_fused__native_batch_norm_legit_no_training_12(c_void_p(buf25.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg191_1
    del arg192_1
    del arg34_1
    del arg35_1
    # Source Nodes: [getattr_l__mod___features___5___conv_0_0], Original ATen: [aten.convolution]
    buf26 = extern_kernels.convolution(buf25, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf26, (4, 192, 28, 28), (150528, 1, 5376, 192))
    del arg36_1
    buf27 = buf26; del buf26  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_13(c_void_p(buf27.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()))
    del arg194_1
    del arg195_1
    del arg37_1
    del arg38_1
    # Source Nodes: [getattr_l__mod___features___5___conv_0_1, getattr_l__mod___features___5___conv_0_2, getattr_l__mod___features___5___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf28 = extern_kernels.convolution(buf27, arg39_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
    assert_size_stride(buf28, (4, 192, 28, 28), (150528, 1, 5376, 192))
    del arg39_1
    del buf27
    buf29 = buf28; del buf28  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_14(c_void_p(buf29.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg197_1
    del arg198_1
    del arg40_1
    del arg41_1
    # Source Nodes: [getattr_l__mod___features___5___conv_1_1, getattr_l__mod___features___5___conv_1_2, getattr_l__mod___features___5___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf30 = extern_kernels.convolution(buf29, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf30, (4, 32, 28, 28), (25088, 1, 896, 32))
    del arg42_1
    del buf29
    buf31 = buf25; del buf25  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_15(c_void_p(buf31.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg44_1.data_ptr()))
    del arg200_1
    del arg201_1
    del arg43_1
    del arg44_1
    del buf30
    # Source Nodes: [getattr_l__mod___features___6___conv_0_0], Original ATen: [aten.convolution]
    buf32 = extern_kernels.convolution(buf31, arg45_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf32, (4, 192, 28, 28), (150528, 1, 5376, 192))
    del arg45_1
    buf33 = buf32; del buf32  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_16(c_void_p(buf33.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg203_1
    del arg204_1
    del arg46_1
    del arg47_1
    # Source Nodes: [getattr_l__mod___features___6___conv_0_1, getattr_l__mod___features___6___conv_0_2, getattr_l__mod___features___6___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf34 = extern_kernels.convolution(buf33, arg48_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
    assert_size_stride(buf34, (4, 192, 28, 28), (150528, 1, 5376, 192))
    del arg48_1
    del buf33
    buf35 = buf34; del buf34  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_17(c_void_p(buf35.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg50_1.data_ptr()))
    del arg206_1
    del arg207_1
    del arg49_1
    del arg50_1
    # Source Nodes: [getattr_l__mod___features___6___conv_1_1, getattr_l__mod___features___6___conv_1_2, getattr_l__mod___features___6___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf36 = extern_kernels.convolution(buf35, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf36, (4, 32, 28, 28), (25088, 1, 896, 32))
    del arg51_1
    del buf35
    buf37 = buf31; del buf31  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_18(c_void_p(buf37.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg209_1
    del arg210_1
    del arg52_1
    del arg53_1
    del buf36
    # Source Nodes: [add_2, getattr_l__mod___features___6___conv_3, getattr_l__mod___features___7___conv_0_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf38 = extern_kernels.convolution(buf37, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf38, (4, 192, 28, 28), (150528, 1, 5376, 192))
    del arg54_1
    del buf37
    buf39 = buf38; del buf38  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_19(c_void_p(buf39.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg56_1.data_ptr()))
    del arg212_1
    del arg213_1
    del arg55_1
    del arg56_1
    # Source Nodes: [getattr_l__mod___features___7___conv_0_1, getattr_l__mod___features___7___conv_0_2, getattr_l__mod___features___7___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf40 = extern_kernels.convolution(buf39, arg57_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
    assert_size_stride(buf40, (4, 192, 14, 14), (37632, 1, 2688, 192))
    del arg57_1
    del buf39
    buf41 = buf40; del buf40  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_20(c_void_p(buf41.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg215_1
    del arg216_1
    del arg58_1
    del arg59_1
    # Source Nodes: [getattr_l__mod___features___7___conv_1_1, getattr_l__mod___features___7___conv_1_2, getattr_l__mod___features___7___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf42 = extern_kernels.convolution(buf41, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf42, (4, 64, 14, 14), (12544, 1, 896, 64))
    del arg60_1
    del buf41
    buf43 = buf42; del buf42  # reuse
    cpp_fused__native_batch_norm_legit_no_training_21(c_void_p(buf43.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()))
    del arg218_1
    del arg219_1
    del arg61_1
    del arg62_1
    # Source Nodes: [getattr_l__mod___features___8___conv_0_0], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf43, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf44, (4, 384, 14, 14), (75264, 1, 5376, 384))
    del arg63_1
    buf45 = buf44; del buf44  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_22(c_void_p(buf45.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg221_1
    del arg222_1
    del arg64_1
    del arg65_1
    # Source Nodes: [getattr_l__mod___features___8___conv_0_1, getattr_l__mod___features___8___conv_0_2, getattr_l__mod___features___8___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf46 = extern_kernels.convolution(buf45, arg66_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
    assert_size_stride(buf46, (4, 384, 14, 14), (75264, 1, 5376, 384))
    del arg66_1
    del buf45
    buf47 = buf46; del buf46  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_23(c_void_p(buf47.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()))
    del arg224_1
    del arg225_1
    del arg67_1
    del arg68_1
    # Source Nodes: [getattr_l__mod___features___8___conv_1_1, getattr_l__mod___features___8___conv_1_2, getattr_l__mod___features___8___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf48 = extern_kernels.convolution(buf47, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf48, (4, 64, 14, 14), (12544, 1, 896, 64))
    del arg69_1
    del buf47
    buf49 = buf43; del buf43  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_24(c_void_p(buf49.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg227_1
    del arg228_1
    del arg70_1
    del arg71_1
    del buf48
    # Source Nodes: [getattr_l__mod___features___9___conv_0_0], Original ATen: [aten.convolution]
    buf50 = extern_kernels.convolution(buf49, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf50, (4, 384, 14, 14), (75264, 1, 5376, 384))
    del arg72_1
    buf51 = buf50; del buf50  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_25(c_void_p(buf51.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()))
    del arg230_1
    del arg231_1
    del arg73_1
    del arg74_1
    # Source Nodes: [getattr_l__mod___features___9___conv_0_1, getattr_l__mod___features___9___conv_0_2, getattr_l__mod___features___9___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf52 = extern_kernels.convolution(buf51, arg75_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
    assert_size_stride(buf52, (4, 384, 14, 14), (75264, 1, 5376, 384))
    del arg75_1
    del buf51
    buf53 = buf52; del buf52  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_26(c_void_p(buf53.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg233_1
    del arg234_1
    del arg76_1
    del arg77_1
    # Source Nodes: [getattr_l__mod___features___9___conv_1_1, getattr_l__mod___features___9___conv_1_2, getattr_l__mod___features___9___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf54 = extern_kernels.convolution(buf53, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf54, (4, 64, 14, 14), (12544, 1, 896, 64))
    del arg78_1
    del buf53
    buf55 = buf49; del buf49  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_27(c_void_p(buf55.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()))
    del arg236_1
    del arg237_1
    del arg79_1
    del arg80_1
    del buf54
    # Source Nodes: [getattr_l__mod___features___10___conv_0_0], Original ATen: [aten.convolution]
    buf56 = extern_kernels.convolution(buf55, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf56, (4, 384, 14, 14), (75264, 1, 5376, 384))
    del arg81_1
    buf57 = buf56; del buf56  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_28(c_void_p(buf57.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg239_1
    del arg240_1
    del arg82_1
    del arg83_1
    # Source Nodes: [getattr_l__mod___features___10___conv_0_1, getattr_l__mod___features___10___conv_0_2, getattr_l__mod___features___10___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf58 = extern_kernels.convolution(buf57, arg84_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
    assert_size_stride(buf58, (4, 384, 14, 14), (75264, 1, 5376, 384))
    del arg84_1
    del buf57
    buf59 = buf58; del buf58  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_29(c_void_p(buf59.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(arg86_1.data_ptr()))
    del arg242_1
    del arg243_1
    del arg85_1
    del arg86_1
    # Source Nodes: [getattr_l__mod___features___10___conv_1_1, getattr_l__mod___features___10___conv_1_2, getattr_l__mod___features___10___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf60 = extern_kernels.convolution(buf59, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf60, (4, 64, 14, 14), (12544, 1, 896, 64))
    del arg87_1
    del buf59
    buf61 = buf55; del buf55  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_30(c_void_p(buf61.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg245_1
    del arg246_1
    del arg88_1
    del arg89_1
    del buf60
    # Source Nodes: [add_5, getattr_l__mod___features___10___conv_3, getattr_l__mod___features___11___conv_0_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf62 = extern_kernels.convolution(buf61, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf62, (4, 384, 14, 14), (75264, 1, 5376, 384))
    del arg90_1
    del buf61
    buf63 = buf62; del buf62  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_31(c_void_p(buf63.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg92_1.data_ptr()))
    del arg248_1
    del arg249_1
    del arg91_1
    del arg92_1
    # Source Nodes: [getattr_l__mod___features___11___conv_0_1, getattr_l__mod___features___11___conv_0_2, getattr_l__mod___features___11___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf64 = extern_kernels.convolution(buf63, arg93_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
    assert_size_stride(buf64, (4, 384, 14, 14), (75264, 1, 5376, 384))
    del arg93_1
    del buf63
    buf65 = buf64; del buf64  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_32(c_void_p(buf65.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg251_1
    del arg252_1
    del arg94_1
    del arg95_1
    # Source Nodes: [getattr_l__mod___features___11___conv_1_1, getattr_l__mod___features___11___conv_1_2, getattr_l__mod___features___11___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf66 = extern_kernels.convolution(buf65, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf66, (4, 96, 14, 14), (18816, 1, 1344, 96))
    del arg96_1
    del buf65
    buf67 = buf66; del buf66  # reuse
    cpp_fused__native_batch_norm_legit_no_training_33(c_void_p(buf67.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg98_1.data_ptr()))
    del arg254_1
    del arg255_1
    del arg97_1
    del arg98_1
    # Source Nodes: [getattr_l__mod___features___12___conv_0_0], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (4, 576, 14, 14), (112896, 1, 8064, 576))
    del arg99_1
    buf69 = buf68; del buf68  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_34(c_void_p(buf69.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    del arg257_1
    del arg258_1
    # Source Nodes: [getattr_l__mod___features___12___conv_0_1, getattr_l__mod___features___12___conv_0_2, getattr_l__mod___features___12___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf70 = extern_kernels.convolution(buf69, arg102_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
    assert_size_stride(buf70, (4, 576, 14, 14), (112896, 1, 8064, 576))
    del arg102_1
    del buf69
    buf71 = buf70; del buf70  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_35(c_void_p(buf71.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg104_1.data_ptr()))
    del arg103_1
    del arg104_1
    del arg260_1
    del arg261_1
    # Source Nodes: [getattr_l__mod___features___12___conv_1_1, getattr_l__mod___features___12___conv_1_2, getattr_l__mod___features___12___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf72 = extern_kernels.convolution(buf71, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf72, (4, 96, 14, 14), (18816, 1, 1344, 96))
    del arg105_1
    del buf71
    buf73 = buf67; del buf67  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_36(c_void_p(buf73.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg263_1
    del arg264_1
    del buf72
    # Source Nodes: [getattr_l__mod___features___13___conv_0_0], Original ATen: [aten.convolution]
    buf74 = extern_kernels.convolution(buf73, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf74, (4, 576, 14, 14), (112896, 1, 8064, 576))
    del arg108_1
    buf75 = buf74; del buf74  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_37(c_void_p(buf75.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()))
    del arg109_1
    del arg110_1
    del arg266_1
    del arg267_1
    # Source Nodes: [getattr_l__mod___features___13___conv_0_1, getattr_l__mod___features___13___conv_0_2, getattr_l__mod___features___13___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf76 = extern_kernels.convolution(buf75, arg111_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
    assert_size_stride(buf76, (4, 576, 14, 14), (112896, 1, 8064, 576))
    del arg111_1
    del buf75
    buf77 = buf76; del buf76  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_38(c_void_p(buf77.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()))
    del arg112_1
    del arg113_1
    del arg269_1
    del arg270_1
    # Source Nodes: [getattr_l__mod___features___13___conv_1_1, getattr_l__mod___features___13___conv_1_2, getattr_l__mod___features___13___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf78 = extern_kernels.convolution(buf77, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf78, (4, 96, 14, 14), (18816, 1, 1344, 96))
    del arg114_1
    del buf77
    buf79 = buf73; del buf73  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_39(c_void_p(buf79.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()))
    del arg115_1
    del arg116_1
    del arg272_1
    del arg273_1
    del buf78
    # Source Nodes: [add_7, getattr_l__mod___features___13___conv_3, getattr_l__mod___features___14___conv_0_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf80 = extern_kernels.convolution(buf79, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf80, (4, 576, 14, 14), (112896, 1, 8064, 576))
    del arg117_1
    del buf79
    buf81 = buf80; del buf80  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_40(c_void_p(buf81.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()))
    del arg118_1
    del arg119_1
    del arg275_1
    del arg276_1
    # Source Nodes: [getattr_l__mod___features___14___conv_0_1, getattr_l__mod___features___14___conv_0_2, getattr_l__mod___features___14___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf82 = extern_kernels.convolution(buf81, arg120_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
    assert_size_stride(buf82, (4, 576, 7, 7), (28224, 1, 4032, 576))
    del arg120_1
    del buf81
    buf83 = buf82; del buf82  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_41(c_void_p(buf83.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()))
    del arg121_1
    del arg122_1
    del arg278_1
    del arg279_1
    # Source Nodes: [getattr_l__mod___features___14___conv_1_1, getattr_l__mod___features___14___conv_1_2, getattr_l__mod___features___14___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf84 = extern_kernels.convolution(buf83, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (4, 160, 7, 7), (7840, 1, 1120, 160))
    del arg123_1
    del buf83
    buf85 = buf84; del buf84  # reuse
    cpp_fused__native_batch_norm_legit_no_training_42(c_void_p(buf85.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()))
    del arg124_1
    del arg125_1
    del arg281_1
    del arg282_1
    # Source Nodes: [getattr_l__mod___features___15___conv_0_0], Original ATen: [aten.convolution]
    buf86 = extern_kernels.convolution(buf85, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf86, (4, 960, 7, 7), (47040, 1, 6720, 960))
    del arg126_1
    buf87 = buf86; del buf86  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_43(c_void_p(buf87.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()))
    del arg127_1
    del arg128_1
    del arg284_1
    del arg285_1
    # Source Nodes: [getattr_l__mod___features___15___conv_0_1, getattr_l__mod___features___15___conv_0_2, getattr_l__mod___features___15___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf88 = extern_kernels.convolution(buf87, arg129_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
    assert_size_stride(buf88, (4, 960, 7, 7), (47040, 1, 6720, 960))
    del arg129_1
    del buf87
    buf89 = buf88; del buf88  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_44(c_void_p(buf89.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()))
    del arg130_1
    del arg131_1
    del arg287_1
    del arg288_1
    # Source Nodes: [getattr_l__mod___features___15___conv_1_1, getattr_l__mod___features___15___conv_1_2, getattr_l__mod___features___15___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf90 = extern_kernels.convolution(buf89, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf90, (4, 160, 7, 7), (7840, 1, 1120, 160))
    del arg132_1
    del buf89
    buf91 = buf85; del buf85  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_45(c_void_p(buf91.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()))
    del arg133_1
    del arg134_1
    del arg290_1
    del arg291_1
    del buf90
    # Source Nodes: [getattr_l__mod___features___16___conv_0_0], Original ATen: [aten.convolution]
    buf92 = extern_kernels.convolution(buf91, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf92, (4, 960, 7, 7), (47040, 1, 6720, 960))
    del arg135_1
    buf93 = buf92; del buf92  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_46(c_void_p(buf93.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()))
    del arg136_1
    del arg137_1
    del arg293_1
    del arg294_1
    # Source Nodes: [getattr_l__mod___features___16___conv_0_1, getattr_l__mod___features___16___conv_0_2, getattr_l__mod___features___16___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf94 = extern_kernels.convolution(buf93, arg138_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
    assert_size_stride(buf94, (4, 960, 7, 7), (47040, 1, 6720, 960))
    del arg138_1
    del buf93
    buf95 = buf94; del buf94  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_47(c_void_p(buf95.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()))
    del arg139_1
    del arg140_1
    del arg296_1
    del arg297_1
    # Source Nodes: [getattr_l__mod___features___16___conv_1_1, getattr_l__mod___features___16___conv_1_2, getattr_l__mod___features___16___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf96 = extern_kernels.convolution(buf95, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf96, (4, 160, 7, 7), (7840, 1, 1120, 160))
    del arg141_1
    del buf95
    buf97 = buf91; del buf91  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_48(c_void_p(buf97.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()))
    del arg142_1
    del arg143_1
    del arg299_1
    del arg300_1
    del buf96
    # Source Nodes: [add_9, getattr_l__mod___features___16___conv_3, getattr_l__mod___features___17___conv_0_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf98 = extern_kernels.convolution(buf97, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf98, (4, 960, 7, 7), (47040, 1, 6720, 960))
    del arg144_1
    del buf97
    buf99 = buf98; del buf98  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_49(c_void_p(buf99.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()))
    del arg145_1
    del arg146_1
    del arg302_1
    del arg303_1
    # Source Nodes: [getattr_l__mod___features___17___conv_0_1, getattr_l__mod___features___17___conv_0_2, getattr_l__mod___features___17___conv_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf100 = extern_kernels.convolution(buf99, arg147_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
    assert_size_stride(buf100, (4, 960, 7, 7), (47040, 1, 6720, 960))
    del arg147_1
    del buf99
    buf101 = buf100; del buf100  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_50(c_void_p(buf101.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()))
    del arg148_1
    del arg149_1
    del arg305_1
    del arg306_1
    # Source Nodes: [getattr_l__mod___features___17___conv_1_1, getattr_l__mod___features___17___conv_1_2, getattr_l__mod___features___17___conv_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf102 = extern_kernels.convolution(buf101, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (4, 320, 7, 7), (15680, 1, 2240, 320))
    del arg150_1
    del buf101
    buf103 = buf102; del buf102  # reuse
    cpp_fused__native_batch_norm_legit_no_training_51(c_void_p(buf103.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()))
    del arg151_1
    del arg152_1
    del arg308_1
    del arg309_1
    # Source Nodes: [getattr_l__mod___features___17___conv_3, l__mod___features_18_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf104 = extern_kernels.convolution(buf103, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf104, (4, 1280, 7, 7), (62720, 1, 8960, 1280))
    del arg153_1
    del buf103
    buf105 = empty_strided((4, 1280, 1, 1), (1280, 1, 5120, 5120), device='cpu', dtype=torch.float32)
    buf106 = reinterpret_tensor(buf105, (4, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf105  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_mean_52(c_void_p(buf106.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()))
    del arg154_1
    del arg155_1
    del arg311_1
    del arg312_1
    del buf104
    buf107 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg157_1, reinterpret_tensor(buf106, (4, 1280), (1280, 1), 0), reinterpret_tensor(arg156_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf107)
    del arg156_1
    del arg157_1
    return (buf107, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((32, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((160, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((320, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg161_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg164_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg167_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg170_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg173_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg176_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg179_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg182_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg185_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg188_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg191_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg194_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg197_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg200_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg203_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg206_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg209_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg212_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg215_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg218_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg221_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg224_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg227_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg230_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg233_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg236_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg239_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg242_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg245_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg248_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg251_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg254_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg257_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg260_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg263_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg266_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg269_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg272_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg275_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg278_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg281_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg284_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg287_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg290_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg293_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg296_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg299_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg302_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg305_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg308_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg311_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg314_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenet_v2', benchmark_compiled_module)
