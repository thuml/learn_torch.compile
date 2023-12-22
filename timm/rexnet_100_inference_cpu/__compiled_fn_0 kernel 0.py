
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_silu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9633792L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (27L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (27L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(24L); x1<static_cast<long>(27L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (27L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (27L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_7 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (162L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (162L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(160L); x1<static_cast<long>(162L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (162L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (162L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4064256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (162L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (162L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(160L); x1<static_cast<long>(162L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (162L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = max_propagate_nan(tmp14, tmp15);
                    auto tmp17 = static_cast<float>(6.0);
                    auto tmp18 = min_propagate_nan(tmp16, tmp17);
                    in_out_ptr0[static_cast<long>(x1 + (162L*x0))] = tmp18;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_9 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(38L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (38L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = c10::convert<long>(x1);
                    auto tmp16 = static_cast<long>(0);
                    auto tmp17 = tmp15 >= tmp16;
                    auto tmp18 = static_cast<long>(27);
                    auto tmp19 = tmp15 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr4[static_cast<long>(x1 + (27L*x0))];
                        auto tmp22 = decltype(tmp14)(tmp14 + tmp21);
                        return tmp22;
                    }
                    ;
                    auto tmp23 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp24 = tmp15 >= tmp18;
                    auto tmp25 = static_cast<long>(38);
                    auto tmp26 = tmp15 < tmp25;
                    auto tmp27 = [&]
                    {
                        return tmp14;
                    }
                    ;
                    auto tmp28 = tmp24 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp29 = tmp19 ? tmp23 : tmp28;
                    in_out_ptr0[static_cast<long>(x1 + (38L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_10 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (228L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (228L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(224L); x1<static_cast<long>(228L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (228L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (228L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(5720064L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (228L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (228L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(224L); x1<static_cast<long>(228L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (228L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (228L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (228L*x2) + (178752L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (228L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(224L); x1<static_cast<long>(228L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (228L*x2) + (178752L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr0[static_cast<long>(x1 + (228L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1824L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (19L*x0)));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (19L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(16L); x1<static_cast<long>(19L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (19L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                in_out_ptr0[static_cast<long>(x1 + (19L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_13 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(224L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (228L*x1) + (178752L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (228L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (228L*x1) + (178752L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(224L); x2<static_cast<long>(228L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (228L*x1) + (178752L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (228L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (228L*x1) + (178752L*x0))] = tmp7;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (50L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (50L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (50L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (50L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_15 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(296L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (300L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (300L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(296L); x1<static_cast<long>(300L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (300L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (300L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1881600L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(296L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (300L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (300L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(296L); x1<static_cast<long>(300L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (300L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (300L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(296L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (300L*x2) + (235200L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (300L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(296L); x1<static_cast<long>(300L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (300L*x2) + (235200L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr0[static_cast<long>(x1 + (300L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2400L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (25L*x0)));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (25L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(24L); x1<static_cast<long>(25L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (25L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                in_out_ptr0[static_cast<long>(x1 + (25L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_18 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(296L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (300L*x1) + (235200L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (300L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (300L*x1) + (235200L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(296L); x2<static_cast<long>(300L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (300L*x1) + (235200L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (300L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (300L*x1) + (235200L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_19 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(61L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (61L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = c10::convert<long>(x1);
                    auto tmp16 = static_cast<long>(0);
                    auto tmp17 = tmp15 >= tmp16;
                    auto tmp18 = static_cast<long>(50);
                    auto tmp19 = tmp15 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr4[static_cast<long>(x1 + (50L*x0))];
                        auto tmp22 = decltype(tmp14)(tmp14 + tmp21);
                        return tmp22;
                    }
                    ;
                    auto tmp23 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp24 = tmp15 >= tmp18;
                    auto tmp25 = static_cast<long>(61);
                    auto tmp26 = tmp15 < tmp25;
                    auto tmp27 = [&]
                    {
                        return tmp14;
                    }
                    ;
                    auto tmp28 = tmp24 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp29 = tmp19 ? tmp23 : tmp28;
                    in_out_ptr0[static_cast<long>(x1 + (61L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_20 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (366L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (366L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(360L); x1<static_cast<long>(366L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (366L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (366L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2295552L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (366L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (366L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(360L); x1<static_cast<long>(366L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (366L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (366L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (366L*x2) + (71736L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (366L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(360L); x1<static_cast<long>(366L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (366L*x2) + (71736L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr0[static_cast<long>(x1 + (366L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2928L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (30L*x0)));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (30L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(24L); x1<static_cast<long>(30L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (30L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                in_out_ptr0[static_cast<long>(x1 + (30L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_23 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (366L*x1) + (71736L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (366L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (366L*x1) + (71736L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(360L); x2<static_cast<long>(366L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (366L*x1) + (71736L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (366L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (366L*x1) + (71736L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_24 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_25 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(432L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (432L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (432L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(677376L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(432L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (432L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (432L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(432L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (432L*x2) + (84672L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (432L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3456L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (36L*x0)));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (36L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(32L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (36L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                in_out_ptr0[static_cast<long>(x1 + (36L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_28 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(432L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (432L*x1) + (84672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (432L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (432L*x1) + (84672L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_29 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(84L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (84L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = c10::convert<long>(x1);
                    auto tmp16 = static_cast<long>(0);
                    auto tmp17 = tmp15 >= tmp16;
                    auto tmp18 = static_cast<long>(72);
                    auto tmp19 = tmp15 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr4[static_cast<long>(x1 + (72L*x0))];
                        auto tmp22 = decltype(tmp14)(tmp14 + tmp21);
                        return tmp22;
                    }
                    ;
                    auto tmp23 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp24 = tmp15 >= tmp18;
                    auto tmp25 = static_cast<long>(84);
                    auto tmp26 = tmp15 < tmp25;
                    auto tmp27 = [&]
                    {
                        return tmp14;
                    }
                    ;
                    auto tmp28 = tmp24 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp29 = tmp19 ? tmp23 : tmp28;
                    in_out_ptr0[static_cast<long>(x1 + (84L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_30 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(504L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (504L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (504L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(790272L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(504L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (504L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (504L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(504L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (504L*x2) + (98784L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (504L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4032L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (42L*x0)));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (42L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(40L); x1<static_cast<long>(42L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (42L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                in_out_ptr0[static_cast<long>(x1 + (42L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_33 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(504L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (504L*x1) + (98784L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (504L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (504L*x1) + (98784L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_34 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(95L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (95L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = c10::convert<long>(x1);
                    auto tmp16 = static_cast<long>(0);
                    auto tmp17 = tmp15 >= tmp16;
                    auto tmp18 = static_cast<long>(84);
                    auto tmp19 = tmp15 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr4[static_cast<long>(x1 + (84L*x0))];
                        auto tmp22 = decltype(tmp14)(tmp14 + tmp21);
                        return tmp22;
                    }
                    ;
                    auto tmp23 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp24 = tmp15 >= tmp18;
                    auto tmp25 = static_cast<long>(95);
                    auto tmp26 = tmp15 < tmp25;
                    auto tmp27 = [&]
                    {
                        return tmp14;
                    }
                    ;
                    auto tmp28 = tmp24 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp29 = tmp19 ? tmp23 : tmp28;
                    in_out_ptr0[static_cast<long>(x1 + (95L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_35 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(568L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (570L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (570L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(568L); x1<static_cast<long>(570L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (570L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (570L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(893760L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(568L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (570L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (570L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(568L); x1<static_cast<long>(570L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (570L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (570L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(568L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (570L*x2) + (111720L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (570L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(568L); x1<static_cast<long>(570L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (570L*x2) + (111720L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr0[static_cast<long>(x1 + (570L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4560L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (47L*x0)));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (47L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(40L); x1<static_cast<long>(47L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (47L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                in_out_ptr0[static_cast<long>(x1 + (47L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_38 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(568L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (570L*x1) + (111720L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (570L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (570L*x1) + (111720L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(568L); x2<static_cast<long>(570L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (570L*x1) + (111720L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (570L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (570L*x1) + (111720L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_39 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(106L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (106L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = c10::convert<long>(x1);
                    auto tmp16 = static_cast<long>(0);
                    auto tmp17 = tmp15 >= tmp16;
                    auto tmp18 = static_cast<long>(95);
                    auto tmp19 = tmp15 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr4[static_cast<long>(x1 + (95L*x0))];
                        auto tmp22 = decltype(tmp14)(tmp14 + tmp21);
                        return tmp22;
                    }
                    ;
                    auto tmp23 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp24 = tmp15 >= tmp18;
                    auto tmp25 = static_cast<long>(106);
                    auto tmp26 = tmp15 < tmp25;
                    auto tmp27 = [&]
                    {
                        return tmp14;
                    }
                    ;
                    auto tmp28 = tmp24 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp29 = tmp19 ? tmp23 : tmp28;
                    in_out_ptr0[static_cast<long>(x1 + (106L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_40 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(632L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (636L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (636L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(632L); x1<static_cast<long>(636L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (636L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (636L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(997248L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(632L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (636L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (636L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(632L); x1<static_cast<long>(636L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (636L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (636L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(632L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (636L*x2) + (124656L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (636L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(632L); x1<static_cast<long>(636L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (636L*x2) + (124656L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr0[static_cast<long>(x1 + (636L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5088L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (53L*x0)));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (53L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(48L); x1<static_cast<long>(53L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (53L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                in_out_ptr0[static_cast<long>(x1 + (53L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_43 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(632L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (636L*x1) + (124656L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (636L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (636L*x1) + (124656L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(632L); x2<static_cast<long>(636L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (636L*x1) + (124656L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (636L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (636L*x1) + (124656L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_44 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(117L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (117L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = c10::convert<long>(x1);
                    auto tmp16 = static_cast<long>(0);
                    auto tmp17 = tmp15 >= tmp16;
                    auto tmp18 = static_cast<long>(106);
                    auto tmp19 = tmp15 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr4[static_cast<long>(x1 + (106L*x0))];
                        auto tmp22 = decltype(tmp14)(tmp14 + tmp21);
                        return tmp22;
                    }
                    ;
                    auto tmp23 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp24 = tmp15 >= tmp18;
                    auto tmp25 = static_cast<long>(117);
                    auto tmp26 = tmp15 < tmp25;
                    auto tmp27 = [&]
                    {
                        return tmp14;
                    }
                    ;
                    auto tmp28 = tmp24 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp29 = tmp19 ? tmp23 : tmp28;
                    in_out_ptr0[static_cast<long>(x1 + (117L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_45 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(696L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (702L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (702L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(696L); x1<static_cast<long>(702L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (702L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (702L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1100736L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(696L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (702L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (702L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(696L); x1<static_cast<long>(702L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (702L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (702L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(696L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (702L*x2) + (137592L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (702L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(696L); x1<static_cast<long>(702L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (702L*x2) + (137592L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr0[static_cast<long>(x1 + (702L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5616L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (58L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                in_out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_48 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(696L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (702L*x1) + (137592L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (702L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (702L*x1) + (137592L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(696L); x2<static_cast<long>(702L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (702L*x1) + (137592L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (702L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (702L*x1) + (137592L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_49 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = c10::convert<long>(x1);
                    auto tmp16 = static_cast<long>(0);
                    auto tmp17 = tmp15 >= tmp16;
                    auto tmp18 = static_cast<long>(117);
                    auto tmp19 = tmp15 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr4[static_cast<long>(x1 + (117L*x0))];
                        auto tmp22 = decltype(tmp14)(tmp14 + tmp21);
                        return tmp22;
                    }
                    ;
                    auto tmp23 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp24 = tmp15 >= tmp18;
                    auto tmp25 = static_cast<long>(128);
                    auto tmp26 = tmp15 < tmp25;
                    auto tmp27 = [&]
                    {
                        return tmp14;
                    }
                    ;
                    auto tmp28 = tmp24 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp29 = tmp19 ? tmp23 : tmp28;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_50 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
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


cpp_fused__native_batch_norm_legit_no_training_relu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_53 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (37632L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (37632L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_54 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(136L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (140L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (140L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(136L); x1<static_cast<long>(140L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (140L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                in_out_ptr0[static_cast<long>(x1 + (140L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_55 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(840L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (840L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (840L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(329280L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(840L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (840L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (840L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(840L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (840L*x2) + (41160L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (840L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6720L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (70L*x0)));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (70L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(64L); x1<static_cast<long>(70L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (70L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                in_out_ptr0[static_cast<long>(x1 + (70L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_58 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(840L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (840L*x1) + (41160L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (840L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (840L*x1) + (41160L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_59 = async_compile.cpp('''
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
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(151L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (151L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = c10::convert<long>(x1);
                auto tmp16 = static_cast<long>(0);
                auto tmp17 = tmp15 >= tmp16;
                auto tmp18 = static_cast<long>(140);
                auto tmp19 = tmp15 < tmp18;
                auto tmp20 = [&]
                {
                    auto tmp21 = in_ptr4[static_cast<long>(x1 + (140L*x0))];
                    auto tmp22 = decltype(tmp14)(tmp14 + tmp21);
                    return tmp22;
                }
                ;
                auto tmp23 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                auto tmp24 = tmp15 >= tmp18;
                auto tmp25 = static_cast<long>(151);
                auto tmp26 = tmp15 < tmp25;
                auto tmp27 = [&]
                {
                    return tmp14;
                }
                ;
                auto tmp28 = tmp24 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                auto tmp29 = tmp19 ? tmp23 : tmp28;
                in_out_ptr0[static_cast<long>(x1 + (151L*x0))] = tmp29;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_60 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(904L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (906L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (906L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(904L); x1<static_cast<long>(906L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (906L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (906L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(355152L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(904L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (906L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (906L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(904L); x1<static_cast<long>(906L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (906L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (906L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(904L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (906L*x2) + (44394L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (906L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(904L); x1<static_cast<long>(906L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (906L*x2) + (44394L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr0[static_cast<long>(x1 + (906L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7248L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (75L*x0)));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (75L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(72L); x1<static_cast<long>(75L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (75L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                in_out_ptr0[static_cast<long>(x1 + (75L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_63 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(904L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (906L*x1) + (44394L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (906L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (906L*x1) + (44394L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(904L); x2<static_cast<long>(906L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (906L*x1) + (44394L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (906L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (906L*x1) + (44394L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_64 = async_compile.cpp('''
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
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(162L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (162L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = c10::convert<long>(x1);
                auto tmp16 = static_cast<long>(0);
                auto tmp17 = tmp15 >= tmp16;
                auto tmp18 = static_cast<long>(151);
                auto tmp19 = tmp15 < tmp18;
                auto tmp20 = [&]
                {
                    auto tmp21 = in_ptr4[static_cast<long>(x1 + (151L*x0))];
                    auto tmp22 = decltype(tmp14)(tmp14 + tmp21);
                    return tmp22;
                }
                ;
                auto tmp23 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                auto tmp24 = tmp15 >= tmp18;
                auto tmp25 = static_cast<long>(162);
                auto tmp26 = tmp15 < tmp25;
                auto tmp27 = [&]
                {
                    return tmp14;
                }
                ;
                auto tmp28 = tmp24 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                auto tmp29 = tmp19 ? tmp23 : tmp28;
                in_out_ptr0[static_cast<long>(x1 + (162L*x0))] = tmp29;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_65 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(968L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (972L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (972L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(968L); x1<static_cast<long>(972L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (972L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (972L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(381024L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(968L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (972L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (972L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(968L); x1<static_cast<long>(972L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (972L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (972L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(968L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (972L*x2) + (47628L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (972L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(968L); x1<static_cast<long>(972L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (972L*x2) + (47628L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr0[static_cast<long>(x1 + (972L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7776L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (81L*x0)));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (81L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(80L); x1<static_cast<long>(81L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (81L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                in_out_ptr0[static_cast<long>(x1 + (81L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_68 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(968L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (972L*x1) + (47628L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (972L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (972L*x1) + (47628L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(968L); x2<static_cast<long>(972L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (972L*x1) + (47628L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (972L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (972L*x1) + (47628L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_69 = async_compile.cpp('''
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
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(174L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (174L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = c10::convert<long>(x1);
                auto tmp16 = static_cast<long>(0);
                auto tmp17 = tmp15 >= tmp16;
                auto tmp18 = static_cast<long>(162);
                auto tmp19 = tmp15 < tmp18;
                auto tmp20 = [&]
                {
                    auto tmp21 = in_ptr4[static_cast<long>(x1 + (162L*x0))];
                    auto tmp22 = decltype(tmp14)(tmp14 + tmp21);
                    return tmp22;
                }
                ;
                auto tmp23 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                auto tmp24 = tmp15 >= tmp18;
                auto tmp25 = static_cast<long>(174);
                auto tmp26 = tmp15 < tmp25;
                auto tmp27 = [&]
                {
                    return tmp14;
                }
                ;
                auto tmp28 = tmp24 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                auto tmp29 = tmp19 ? tmp23 : tmp28;
                in_out_ptr0[static_cast<long>(x1 + (174L*x0))] = tmp29;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_70 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1040L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1044L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1044L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(1040L); x1<static_cast<long>(1044L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1044L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (1044L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(409248L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_mean_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1040L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1044L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1044L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(1040L); x1<static_cast<long>(1044L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1044L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    in_out_ptr0[static_cast<long>(x1 + (1044L*x0))] = tmp14;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1040L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1044L*x2) + (51156L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1044L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(1040L); x1<static_cast<long>(1044L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1044L*x2) + (51156L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr0[static_cast<long>(x1 + (1044L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8352L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (87L*x0)));
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
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (87L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(80L); x1<static_cast<long>(87L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (87L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                in_out_ptr0[static_cast<long>(x1 + (87L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_73 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1040L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1044L*x1) + (51156L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1044L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (1044L*x1) + (51156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(1040L); x2<static_cast<long>(1044L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1044L*x1) + (51156L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (1044L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (1044L*x1) + (51156L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_74 = async_compile.cpp('''
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
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(185L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (185L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = c10::convert<long>(x1);
                auto tmp16 = static_cast<long>(0);
                auto tmp17 = tmp15 >= tmp16;
                auto tmp18 = static_cast<long>(174);
                auto tmp19 = tmp15 < tmp18;
                auto tmp20 = [&]
                {
                    auto tmp21 = in_ptr4[static_cast<long>(x1 + (174L*x0))];
                    auto tmp22 = decltype(tmp14)(tmp14 + tmp21);
                    return tmp22;
                }
                ;
                auto tmp23 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                auto tmp24 = tmp15 >= tmp18;
                auto tmp25 = static_cast<long>(185);
                auto tmp26 = tmp15 < tmp25;
                auto tmp27 = [&]
                {
                    return tmp14;
                }
                ;
                auto tmp28 = tmp24 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                auto tmp29 = tmp19 ? tmp23 : tmp28;
                in_out_ptr0[static_cast<long>(x1 + (185L*x0))] = tmp29;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1280L*x2) + (62720L*x0)));
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1 = args
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
    assert_size_stride(arg10_1, (27, ), (1, ))
    assert_size_stride(arg11_1, (27, ), (1, ))
    assert_size_stride(arg12_1, (162, ), (1, ))
    assert_size_stride(arg13_1, (162, ), (1, ))
    assert_size_stride(arg14_1, (162, ), (1, ))
    assert_size_stride(arg15_1, (162, ), (1, ))
    assert_size_stride(arg16_1, (38, ), (1, ))
    assert_size_stride(arg17_1, (38, ), (1, ))
    assert_size_stride(arg18_1, (228, ), (1, ))
    assert_size_stride(arg19_1, (228, ), (1, ))
    assert_size_stride(arg20_1, (228, ), (1, ))
    assert_size_stride(arg21_1, (228, ), (1, ))
    assert_size_stride(arg22_1, (50, ), (1, ))
    assert_size_stride(arg23_1, (50, ), (1, ))
    assert_size_stride(arg24_1, (300, ), (1, ))
    assert_size_stride(arg25_1, (300, ), (1, ))
    assert_size_stride(arg26_1, (300, ), (1, ))
    assert_size_stride(arg27_1, (300, ), (1, ))
    assert_size_stride(arg28_1, (61, ), (1, ))
    assert_size_stride(arg29_1, (61, ), (1, ))
    assert_size_stride(arg30_1, (366, ), (1, ))
    assert_size_stride(arg31_1, (366, ), (1, ))
    assert_size_stride(arg32_1, (366, ), (1, ))
    assert_size_stride(arg33_1, (366, ), (1, ))
    assert_size_stride(arg34_1, (72, ), (1, ))
    assert_size_stride(arg35_1, (72, ), (1, ))
    assert_size_stride(arg36_1, (432, ), (1, ))
    assert_size_stride(arg37_1, (432, ), (1, ))
    assert_size_stride(arg38_1, (432, ), (1, ))
    assert_size_stride(arg39_1, (432, ), (1, ))
    assert_size_stride(arg40_1, (84, ), (1, ))
    assert_size_stride(arg41_1, (84, ), (1, ))
    assert_size_stride(arg42_1, (504, ), (1, ))
    assert_size_stride(arg43_1, (504, ), (1, ))
    assert_size_stride(arg44_1, (504, ), (1, ))
    assert_size_stride(arg45_1, (504, ), (1, ))
    assert_size_stride(arg46_1, (95, ), (1, ))
    assert_size_stride(arg47_1, (95, ), (1, ))
    assert_size_stride(arg48_1, (570, ), (1, ))
    assert_size_stride(arg49_1, (570, ), (1, ))
    assert_size_stride(arg50_1, (570, ), (1, ))
    assert_size_stride(arg51_1, (570, ), (1, ))
    assert_size_stride(arg52_1, (106, ), (1, ))
    assert_size_stride(arg53_1, (106, ), (1, ))
    assert_size_stride(arg54_1, (636, ), (1, ))
    assert_size_stride(arg55_1, (636, ), (1, ))
    assert_size_stride(arg56_1, (636, ), (1, ))
    assert_size_stride(arg57_1, (636, ), (1, ))
    assert_size_stride(arg58_1, (117, ), (1, ))
    assert_size_stride(arg59_1, (117, ), (1, ))
    assert_size_stride(arg60_1, (702, ), (1, ))
    assert_size_stride(arg61_1, (702, ), (1, ))
    assert_size_stride(arg62_1, (702, ), (1, ))
    assert_size_stride(arg63_1, (702, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (140, ), (1, ))
    assert_size_stride(arg71_1, (140, ), (1, ))
    assert_size_stride(arg72_1, (840, ), (1, ))
    assert_size_stride(arg73_1, (840, ), (1, ))
    assert_size_stride(arg74_1, (840, ), (1, ))
    assert_size_stride(arg75_1, (840, ), (1, ))
    assert_size_stride(arg76_1, (151, ), (1, ))
    assert_size_stride(arg77_1, (151, ), (1, ))
    assert_size_stride(arg78_1, (906, ), (1, ))
    assert_size_stride(arg79_1, (906, ), (1, ))
    assert_size_stride(arg80_1, (906, ), (1, ))
    assert_size_stride(arg81_1, (906, ), (1, ))
    assert_size_stride(arg82_1, (162, ), (1, ))
    assert_size_stride(arg83_1, (162, ), (1, ))
    assert_size_stride(arg84_1, (972, ), (1, ))
    assert_size_stride(arg85_1, (972, ), (1, ))
    assert_size_stride(arg86_1, (972, ), (1, ))
    assert_size_stride(arg87_1, (972, ), (1, ))
    assert_size_stride(arg88_1, (174, ), (1, ))
    assert_size_stride(arg89_1, (174, ), (1, ))
    assert_size_stride(arg90_1, (1044, ), (1, ))
    assert_size_stride(arg91_1, (1044, ), (1, ))
    assert_size_stride(arg92_1, (1044, ), (1, ))
    assert_size_stride(arg93_1, (1044, ), (1, ))
    assert_size_stride(arg94_1, (185, ), (1, ))
    assert_size_stride(arg95_1, (185, ), (1, ))
    assert_size_stride(arg96_1, (1280, ), (1, ))
    assert_size_stride(arg97_1, (1280, ), (1, ))
    assert_size_stride(arg98_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg99_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg100_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg101_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg102_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg103_1, (27, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg104_1, (162, 27, 1, 1), (27, 1, 1, 1))
    assert_size_stride(arg105_1, (162, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg106_1, (38, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(arg107_1, (228, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg108_1, (228, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg109_1, (19, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(arg110_1, (19, ), (1, ))
    assert_size_stride(arg111_1, (19, ), (1, ))
    assert_size_stride(arg112_1, (19, ), (1, ))
    assert_size_stride(arg113_1, (228, 19, 1, 1), (19, 1, 1, 1))
    assert_size_stride(arg114_1, (228, ), (1, ))
    assert_size_stride(arg115_1, (50, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(arg116_1, (300, 50, 1, 1), (50, 1, 1, 1))
    assert_size_stride(arg117_1, (300, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (25, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(arg119_1, (25, ), (1, ))
    assert_size_stride(arg120_1, (25, ), (1, ))
    assert_size_stride(arg121_1, (25, ), (1, ))
    assert_size_stride(arg122_1, (300, 25, 1, 1), (25, 1, 1, 1))
    assert_size_stride(arg123_1, (300, ), (1, ))
    assert_size_stride(arg124_1, (61, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(arg125_1, (366, 61, 1, 1), (61, 1, 1, 1))
    assert_size_stride(arg126_1, (366, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg127_1, (30, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(arg128_1, (30, ), (1, ))
    assert_size_stride(arg129_1, (30, ), (1, ))
    assert_size_stride(arg130_1, (30, ), (1, ))
    assert_size_stride(arg131_1, (366, 30, 1, 1), (30, 1, 1, 1))
    assert_size_stride(arg132_1, (366, ), (1, ))
    assert_size_stride(arg133_1, (72, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(arg134_1, (432, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg135_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg136_1, (36, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg137_1, (36, ), (1, ))
    assert_size_stride(arg138_1, (36, ), (1, ))
    assert_size_stride(arg139_1, (36, ), (1, ))
    assert_size_stride(arg140_1, (432, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg141_1, (432, ), (1, ))
    assert_size_stride(arg142_1, (84, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg143_1, (504, 84, 1, 1), (84, 1, 1, 1))
    assert_size_stride(arg144_1, (504, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg145_1, (42, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(arg146_1, (42, ), (1, ))
    assert_size_stride(arg147_1, (42, ), (1, ))
    assert_size_stride(arg148_1, (42, ), (1, ))
    assert_size_stride(arg149_1, (504, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(arg150_1, (504, ), (1, ))
    assert_size_stride(arg151_1, (95, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(arg152_1, (570, 95, 1, 1), (95, 1, 1, 1))
    assert_size_stride(arg153_1, (570, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg154_1, (47, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(arg155_1, (47, ), (1, ))
    assert_size_stride(arg156_1, (47, ), (1, ))
    assert_size_stride(arg157_1, (47, ), (1, ))
    assert_size_stride(arg158_1, (570, 47, 1, 1), (47, 1, 1, 1))
    assert_size_stride(arg159_1, (570, ), (1, ))
    assert_size_stride(arg160_1, (106, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(arg161_1, (636, 106, 1, 1), (106, 1, 1, 1))
    assert_size_stride(arg162_1, (636, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg163_1, (53, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(arg164_1, (53, ), (1, ))
    assert_size_stride(arg165_1, (53, ), (1, ))
    assert_size_stride(arg166_1, (53, ), (1, ))
    assert_size_stride(arg167_1, (636, 53, 1, 1), (53, 1, 1, 1))
    assert_size_stride(arg168_1, (636, ), (1, ))
    assert_size_stride(arg169_1, (117, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(arg170_1, (702, 117, 1, 1), (117, 1, 1, 1))
    assert_size_stride(arg171_1, (702, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg172_1, (58, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(arg173_1, (58, ), (1, ))
    assert_size_stride(arg174_1, (58, ), (1, ))
    assert_size_stride(arg175_1, (58, ), (1, ))
    assert_size_stride(arg176_1, (702, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg177_1, (702, ), (1, ))
    assert_size_stride(arg178_1, (128, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(arg179_1, (768, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg180_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg181_1, (64, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg182_1, (64, ), (1, ))
    assert_size_stride(arg183_1, (64, ), (1, ))
    assert_size_stride(arg184_1, (64, ), (1, ))
    assert_size_stride(arg185_1, (768, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (140, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg188_1, (840, 140, 1, 1), (140, 1, 1, 1))
    assert_size_stride(arg189_1, (840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg190_1, (70, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(arg191_1, (70, ), (1, ))
    assert_size_stride(arg192_1, (70, ), (1, ))
    assert_size_stride(arg193_1, (70, ), (1, ))
    assert_size_stride(arg194_1, (840, 70, 1, 1), (70, 1, 1, 1))
    assert_size_stride(arg195_1, (840, ), (1, ))
    assert_size_stride(arg196_1, (151, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(arg197_1, (906, 151, 1, 1), (151, 1, 1, 1))
    assert_size_stride(arg198_1, (906, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg199_1, (75, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(arg200_1, (75, ), (1, ))
    assert_size_stride(arg201_1, (75, ), (1, ))
    assert_size_stride(arg202_1, (75, ), (1, ))
    assert_size_stride(arg203_1, (906, 75, 1, 1), (75, 1, 1, 1))
    assert_size_stride(arg204_1, (906, ), (1, ))
    assert_size_stride(arg205_1, (162, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(arg206_1, (972, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(arg207_1, (972, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg208_1, (81, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(arg209_1, (81, ), (1, ))
    assert_size_stride(arg210_1, (81, ), (1, ))
    assert_size_stride(arg211_1, (81, ), (1, ))
    assert_size_stride(arg212_1, (972, 81, 1, 1), (81, 1, 1, 1))
    assert_size_stride(arg213_1, (972, ), (1, ))
    assert_size_stride(arg214_1, (174, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(arg215_1, (1044, 174, 1, 1), (174, 1, 1, 1))
    assert_size_stride(arg216_1, (1044, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg217_1, (87, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(arg218_1, (87, ), (1, ))
    assert_size_stride(arg219_1, (87, ), (1, ))
    assert_size_stride(arg220_1, (87, ), (1, ))
    assert_size_stride(arg221_1, (1044, 87, 1, 1), (87, 1, 1, 1))
    assert_size_stride(arg222_1, (1044, ), (1, ))
    assert_size_stride(arg223_1, (185, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(arg224_1, (1280, 185, 1, 1), (185, 1, 1, 1))
    assert_size_stride(arg225_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg226_1, (1000, ), (1, ))
    assert_size_stride(arg227_1, (32, ), (1, ))
    assert_size_stride(arg228_1, (32, ), (1, ))
    assert_size_stride(arg229_1, (32, ), (1, ))
    assert_size_stride(arg230_1, (32, ), (1, ))
    assert_size_stride(arg231_1, (16, ), (1, ))
    assert_size_stride(arg232_1, (16, ), (1, ))
    assert_size_stride(arg233_1, (96, ), (1, ))
    assert_size_stride(arg234_1, (96, ), (1, ))
    assert_size_stride(arg235_1, (96, ), (1, ))
    assert_size_stride(arg236_1, (96, ), (1, ))
    assert_size_stride(arg237_1, (27, ), (1, ))
    assert_size_stride(arg238_1, (27, ), (1, ))
    assert_size_stride(arg239_1, (162, ), (1, ))
    assert_size_stride(arg240_1, (162, ), (1, ))
    assert_size_stride(arg241_1, (162, ), (1, ))
    assert_size_stride(arg242_1, (162, ), (1, ))
    assert_size_stride(arg243_1, (38, ), (1, ))
    assert_size_stride(arg244_1, (38, ), (1, ))
    assert_size_stride(arg245_1, (228, ), (1, ))
    assert_size_stride(arg246_1, (228, ), (1, ))
    assert_size_stride(arg247_1, (228, ), (1, ))
    assert_size_stride(arg248_1, (228, ), (1, ))
    assert_size_stride(arg249_1, (50, ), (1, ))
    assert_size_stride(arg250_1, (50, ), (1, ))
    assert_size_stride(arg251_1, (300, ), (1, ))
    assert_size_stride(arg252_1, (300, ), (1, ))
    assert_size_stride(arg253_1, (300, ), (1, ))
    assert_size_stride(arg254_1, (300, ), (1, ))
    assert_size_stride(arg255_1, (61, ), (1, ))
    assert_size_stride(arg256_1, (61, ), (1, ))
    assert_size_stride(arg257_1, (366, ), (1, ))
    assert_size_stride(arg258_1, (366, ), (1, ))
    assert_size_stride(arg259_1, (366, ), (1, ))
    assert_size_stride(arg260_1, (366, ), (1, ))
    assert_size_stride(arg261_1, (72, ), (1, ))
    assert_size_stride(arg262_1, (72, ), (1, ))
    assert_size_stride(arg263_1, (432, ), (1, ))
    assert_size_stride(arg264_1, (432, ), (1, ))
    assert_size_stride(arg265_1, (432, ), (1, ))
    assert_size_stride(arg266_1, (432, ), (1, ))
    assert_size_stride(arg267_1, (84, ), (1, ))
    assert_size_stride(arg268_1, (84, ), (1, ))
    assert_size_stride(arg269_1, (504, ), (1, ))
    assert_size_stride(arg270_1, (504, ), (1, ))
    assert_size_stride(arg271_1, (504, ), (1, ))
    assert_size_stride(arg272_1, (504, ), (1, ))
    assert_size_stride(arg273_1, (95, ), (1, ))
    assert_size_stride(arg274_1, (95, ), (1, ))
    assert_size_stride(arg275_1, (570, ), (1, ))
    assert_size_stride(arg276_1, (570, ), (1, ))
    assert_size_stride(arg277_1, (570, ), (1, ))
    assert_size_stride(arg278_1, (570, ), (1, ))
    assert_size_stride(arg279_1, (106, ), (1, ))
    assert_size_stride(arg280_1, (106, ), (1, ))
    assert_size_stride(arg281_1, (636, ), (1, ))
    assert_size_stride(arg282_1, (636, ), (1, ))
    assert_size_stride(arg283_1, (636, ), (1, ))
    assert_size_stride(arg284_1, (636, ), (1, ))
    assert_size_stride(arg285_1, (117, ), (1, ))
    assert_size_stride(arg286_1, (117, ), (1, ))
    assert_size_stride(arg287_1, (702, ), (1, ))
    assert_size_stride(arg288_1, (702, ), (1, ))
    assert_size_stride(arg289_1, (702, ), (1, ))
    assert_size_stride(arg290_1, (702, ), (1, ))
    assert_size_stride(arg291_1, (128, ), (1, ))
    assert_size_stride(arg292_1, (128, ), (1, ))
    assert_size_stride(arg293_1, (768, ), (1, ))
    assert_size_stride(arg294_1, (768, ), (1, ))
    assert_size_stride(arg295_1, (768, ), (1, ))
    assert_size_stride(arg296_1, (768, ), (1, ))
    assert_size_stride(arg297_1, (140, ), (1, ))
    assert_size_stride(arg298_1, (140, ), (1, ))
    assert_size_stride(arg299_1, (840, ), (1, ))
    assert_size_stride(arg300_1, (840, ), (1, ))
    assert_size_stride(arg301_1, (840, ), (1, ))
    assert_size_stride(arg302_1, (840, ), (1, ))
    assert_size_stride(arg303_1, (151, ), (1, ))
    assert_size_stride(arg304_1, (151, ), (1, ))
    assert_size_stride(arg305_1, (906, ), (1, ))
    assert_size_stride(arg306_1, (906, ), (1, ))
    assert_size_stride(arg307_1, (906, ), (1, ))
    assert_size_stride(arg308_1, (906, ), (1, ))
    assert_size_stride(arg309_1, (162, ), (1, ))
    assert_size_stride(arg310_1, (162, ), (1, ))
    assert_size_stride(arg311_1, (972, ), (1, ))
    assert_size_stride(arg312_1, (972, ), (1, ))
    assert_size_stride(arg313_1, (972, ), (1, ))
    assert_size_stride(arg314_1, (972, ), (1, ))
    assert_size_stride(arg315_1, (174, ), (1, ))
    assert_size_stride(arg316_1, (174, ), (1, ))
    assert_size_stride(arg317_1, (1044, ), (1, ))
    assert_size_stride(arg318_1, (1044, ), (1, ))
    assert_size_stride(arg319_1, (1044, ), (1, ))
    assert_size_stride(arg320_1, (1044, ), (1, ))
    assert_size_stride(arg321_1, (185, ), (1, ))
    assert_size_stride(arg322_1, (185, ), (1, ))
    assert_size_stride(arg323_1, (1280, ), (1, ))
    assert_size_stride(arg324_1, (1280, ), (1, ))
    assert_size_stride(arg325_1, (19, ), (1, ))
    assert_size_stride(arg326_1, (19, ), (1, ))
    assert_size_stride(arg327_1, (), ())
    assert_size_stride(arg328_1, (25, ), (1, ))
    assert_size_stride(arg329_1, (25, ), (1, ))
    assert_size_stride(arg330_1, (), ())
    assert_size_stride(arg331_1, (30, ), (1, ))
    assert_size_stride(arg332_1, (30, ), (1, ))
    assert_size_stride(arg333_1, (), ())
    assert_size_stride(arg334_1, (36, ), (1, ))
    assert_size_stride(arg335_1, (36, ), (1, ))
    assert_size_stride(arg336_1, (), ())
    assert_size_stride(arg337_1, (42, ), (1, ))
    assert_size_stride(arg338_1, (42, ), (1, ))
    assert_size_stride(arg339_1, (), ())
    assert_size_stride(arg340_1, (47, ), (1, ))
    assert_size_stride(arg341_1, (47, ), (1, ))
    assert_size_stride(arg342_1, (), ())
    assert_size_stride(arg343_1, (53, ), (1, ))
    assert_size_stride(arg344_1, (53, ), (1, ))
    assert_size_stride(arg345_1, (), ())
    assert_size_stride(arg346_1, (58, ), (1, ))
    assert_size_stride(arg347_1, (58, ), (1, ))
    assert_size_stride(arg348_1, (), ())
    assert_size_stride(arg349_1, (64, ), (1, ))
    assert_size_stride(arg350_1, (64, ), (1, ))
    assert_size_stride(arg351_1, (), ())
    assert_size_stride(arg352_1, (70, ), (1, ))
    assert_size_stride(arg353_1, (70, ), (1, ))
    assert_size_stride(arg354_1, (), ())
    assert_size_stride(arg355_1, (75, ), (1, ))
    assert_size_stride(arg356_1, (75, ), (1, ))
    assert_size_stride(arg357_1, (), ())
    assert_size_stride(arg358_1, (81, ), (1, ))
    assert_size_stride(arg359_1, (81, ), (1, ))
    assert_size_stride(arg360_1, (), ())
    assert_size_stride(arg361_1, (87, ), (1, ))
    assert_size_stride(arg362_1, (87, ), (1, ))
    assert_size_stride(arg363_1, (), ())
    assert_size_stride(arg364_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg364_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg364_1
    del arg98_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = buf3; del buf3  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_1(c_void_p(buf4.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg0_1
    del arg1_1
    del arg227_1
    del arg228_1
    # Source Nodes: [shortcut, x_6], Original ATen: [aten.convolution, aten.silu]
    buf5 = extern_kernels.convolution(buf4, arg99_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf5, (8, 32, 112, 112), (401408, 1, 3584, 32))
    del arg99_1
    del buf4
    buf6 = buf5; del buf5  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_2(c_void_p(buf6.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()))
    del arg229_1
    del arg230_1
    del arg2_1
    del arg3_1
    # Source Nodes: [x_12, x_13, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf7 = extern_kernels.convolution(buf6, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf7, (8, 16, 112, 112), (200704, 1, 1792, 16))
    del arg100_1
    del buf6
    buf8 = buf7; del buf7  # reuse
    cpp_fused__native_batch_norm_legit_no_training_3(c_void_p(buf8.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg231_1
    del arg232_1
    del arg4_1
    del arg5_1
    # Source Nodes: [x_14, x_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf9 = extern_kernels.convolution(buf8, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf9, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    del arg101_1
    del buf8
    buf10 = buf9; del buf9  # reuse
    buf11 = buf10; del buf10  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_4(c_void_p(buf11.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()))
    del arg233_1
    del arg234_1
    del arg6_1
    del arg7_1
    # Source Nodes: [x_24, x_25], Original ATen: [aten.convolution, aten.silu]
    buf12 = extern_kernels.convolution(buf11, arg102_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
    assert_size_stride(buf12, (8, 96, 56, 56), (301056, 1, 5376, 96))
    del arg102_1
    del buf11
    buf13 = buf12; del buf12  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_5(c_void_p(buf13.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg235_1
    del arg236_1
    del arg8_1
    del arg9_1
    # Source Nodes: [x_26, x_31, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf14 = extern_kernels.convolution(buf13, arg103_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf14, (8, 27, 56, 56), (84672, 1, 1512, 27))
    del arg103_1
    del buf13
    buf15 = buf14; del buf14  # reuse
    cpp_fused__native_batch_norm_legit_no_training_6(c_void_p(buf15.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()))
    del arg10_1
    del arg11_1
    del arg237_1
    del arg238_1
    # Source Nodes: [x_38], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf15, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf16, (8, 162, 56, 56), (508032, 1, 9072, 162))
    del arg104_1
    buf17 = buf16; del buf16  # reuse
    buf18 = buf17; del buf17  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_7(c_void_p(buf18.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()))
    del arg12_1
    del arg13_1
    del arg239_1
    del arg240_1
    # Source Nodes: [x_43, x_44], Original ATen: [aten.convolution, aten.silu]
    buf19 = extern_kernels.convolution(buf18, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=162, bias=None)
    assert_size_stride(buf19, (8, 162, 56, 56), (508032, 1, 9072, 162))
    del arg105_1
    del buf18
    buf20 = buf19; del buf19  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardtanh_8(c_void_p(buf20.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()))
    del arg14_1
    del arg15_1
    del arg241_1
    del arg242_1
    # Source Nodes: [x_45, x_50, x_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.hardtanh]
    buf21 = extern_kernels.convolution(buf20, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf21, (8, 38, 56, 56), (119168, 1, 2128, 38))
    del arg106_1
    del buf20
    buf22 = buf21; del buf21  # reuse
    buf23 = buf22; del buf22  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_9(c_void_p(buf23.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf15.data_ptr()))
    del arg16_1
    del arg17_1
    del arg243_1
    del arg244_1
    del buf15
    # Source Nodes: [cat_21, x_58], Original ATen: [aten.cat, aten.convolution]
    buf24 = extern_kernels.convolution(buf23, arg107_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf24, (8, 228, 56, 56), (715008, 1, 12768, 228))
    del arg107_1
    del buf23
    buf25 = buf24; del buf24  # reuse
    buf26 = buf25; del buf25  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_10(c_void_p(buf26.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()))
    del arg18_1
    del arg19_1
    del arg245_1
    del arg246_1
    # Source Nodes: [x_63, x_64], Original ATen: [aten.convolution, aten.silu]
    buf27 = extern_kernels.convolution(buf26, arg108_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=228, bias=None)
    assert_size_stride(buf27, (8, 228, 28, 28), (178752, 1, 6384, 228))
    del arg108_1
    del buf26
    buf28 = buf27; del buf27  # reuse
    buf29 = empty_strided((8, 228, 1, 1), (228, 1, 1824, 1824), device='cpu', dtype=torch.float32)
    buf30 = reinterpret_tensor(buf29, (8, 228, 1, 1), (228, 1, 228, 228), 0); del buf29  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_11(c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()))
    del arg20_1
    del arg21_1
    del arg247_1
    del arg248_1
    # Source Nodes: [x_se, x_se_1], Original ATen: [aten.convolution, aten.mean]
    buf31 = extern_kernels.convolution(buf30, arg109_1, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf31, (8, 19, 1, 1), (19, 1, 19, 19))
    del arg109_1
    del arg110_1
    del buf30
    buf32 = buf31; del buf31  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_12(c_void_p(buf32.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg112_1.data_ptr()))
    del arg111_1
    del arg112_1
    del arg325_1
    del arg326_1
    # Source Nodes: [getattr_l__mod___features___3___se_bn, x_se_2, x_se_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf33 = extern_kernels.convolution(buf32, arg113_1, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf33, (8, 228, 1, 1), (228, 1, 228, 228))
    del arg113_1
    del arg114_1
    del buf32
    buf34 = buf28; del buf28  # reuse
    cpp_fused_hardtanh_mul_sigmoid_13(c_void_p(buf34.data_ptr()), c_void_p(buf33.data_ptr()))
    del buf33
    # Source Nodes: [sigmoid, x_70, x_71, x_72], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf35 = extern_kernels.convolution(buf34, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf35, (8, 50, 28, 28), (39200, 1, 1400, 50))
    del arg115_1
    del buf34
    buf36 = buf35; del buf35  # reuse
    cpp_fused__native_batch_norm_legit_no_training_14(c_void_p(buf36.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg22_1
    del arg23_1
    del arg249_1
    del arg250_1
    # Source Nodes: [x_78], Original ATen: [aten.convolution]
    buf37 = extern_kernels.convolution(buf36, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf37, (8, 300, 28, 28), (235200, 1, 8400, 300))
    del arg116_1
    buf38 = buf37; del buf37  # reuse
    buf39 = buf38; del buf38  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_15(c_void_p(buf39.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()))
    del arg24_1
    del arg251_1
    del arg252_1
    del arg25_1
    # Source Nodes: [x_83, x_84], Original ATen: [aten.convolution, aten.silu]
    buf40 = extern_kernels.convolution(buf39, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=300, bias=None)
    assert_size_stride(buf40, (8, 300, 28, 28), (235200, 1, 8400, 300))
    del arg117_1
    del buf39
    buf41 = buf40; del buf40  # reuse
    buf42 = empty_strided((8, 300, 1, 1), (300, 1, 2400, 2400), device='cpu', dtype=torch.float32)
    buf43 = reinterpret_tensor(buf42, (8, 300, 1, 1), (300, 1, 300, 300), 0); del buf42  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_16(c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()))
    del arg253_1
    del arg254_1
    del arg26_1
    del arg27_1
    # Source Nodes: [x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean]
    buf44 = extern_kernels.convolution(buf43, arg118_1, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf44, (8, 25, 1, 1), (25, 1, 25, 25))
    del arg118_1
    del arg119_1
    del buf43
    buf45 = buf44; del buf44  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_17(c_void_p(buf45.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()))
    del arg120_1
    del arg121_1
    del arg328_1
    del arg329_1
    # Source Nodes: [getattr_l__mod___features___4___se_bn, x_se_6, x_se_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf46 = extern_kernels.convolution(buf45, arg122_1, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf46, (8, 300, 1, 1), (300, 1, 300, 300))
    del arg122_1
    del arg123_1
    del buf45
    buf47 = buf41; del buf41  # reuse
    cpp_fused_hardtanh_mul_sigmoid_18(c_void_p(buf47.data_ptr()), c_void_p(buf46.data_ptr()))
    del buf46
    # Source Nodes: [sigmoid_1, x_90, x_91, x_92], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf48 = extern_kernels.convolution(buf47, arg124_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf48, (8, 61, 28, 28), (47824, 1, 1708, 61))
    del arg124_1
    del buf47
    buf49 = buf48; del buf48  # reuse
    buf50 = buf49; del buf49  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_19(c_void_p(buf50.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf36.data_ptr()))
    del arg255_1
    del arg256_1
    del arg28_1
    del arg29_1
    del buf36
    # Source Nodes: [cat_20, x_99], Original ATen: [aten.cat, aten.convolution]
    buf51 = extern_kernels.convolution(buf50, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf51, (8, 366, 28, 28), (286944, 1, 10248, 366))
    del arg125_1
    del buf50
    buf52 = buf51; del buf51  # reuse
    buf53 = buf52; del buf52  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_20(c_void_p(buf53.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()))
    del arg257_1
    del arg258_1
    del arg30_1
    del arg31_1
    # Source Nodes: [x_104, x_105], Original ATen: [aten.convolution, aten.silu]
    buf54 = extern_kernels.convolution(buf53, arg126_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=366, bias=None)
    assert_size_stride(buf54, (8, 366, 14, 14), (71736, 1, 5124, 366))
    del arg126_1
    del buf53
    buf55 = buf54; del buf54  # reuse
    buf56 = empty_strided((8, 366, 1, 1), (366, 1, 2928, 2928), device='cpu', dtype=torch.float32)
    buf57 = reinterpret_tensor(buf56, (8, 366, 1, 1), (366, 1, 366, 366), 0); del buf56  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_21(c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()))
    del arg259_1
    del arg260_1
    del arg32_1
    del arg33_1
    # Source Nodes: [x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean]
    buf58 = extern_kernels.convolution(buf57, arg127_1, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf58, (8, 30, 1, 1), (30, 1, 30, 30))
    del arg127_1
    del arg128_1
    del buf57
    buf59 = buf58; del buf58  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_22(c_void_p(buf59.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(arg130_1.data_ptr()))
    del arg129_1
    del arg130_1
    del arg331_1
    del arg332_1
    # Source Nodes: [getattr_l__mod___features___5___se_bn, x_se_10, x_se_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf60 = extern_kernels.convolution(buf59, arg131_1, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf60, (8, 366, 1, 1), (366, 1, 366, 366))
    del arg131_1
    del arg132_1
    del buf59
    buf61 = buf55; del buf55  # reuse
    cpp_fused_hardtanh_mul_sigmoid_23(c_void_p(buf61.data_ptr()), c_void_p(buf60.data_ptr()))
    del buf60
    # Source Nodes: [sigmoid_2, x_111, x_112, x_113], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf62 = extern_kernels.convolution(buf61, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf62, (8, 72, 14, 14), (14112, 1, 1008, 72))
    del arg133_1
    del buf61
    buf63 = buf62; del buf62  # reuse
    cpp_fused__native_batch_norm_legit_no_training_24(c_void_p(buf63.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg261_1
    del arg262_1
    del arg34_1
    del arg35_1
    # Source Nodes: [x_119], Original ATen: [aten.convolution]
    buf64 = extern_kernels.convolution(buf63, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf64, (8, 432, 14, 14), (84672, 1, 6048, 432))
    del arg134_1
    buf65 = buf64; del buf64  # reuse
    buf66 = buf65; del buf65  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_25(c_void_p(buf66.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()))
    del arg263_1
    del arg264_1
    del arg36_1
    del arg37_1
    # Source Nodes: [x_124, x_125], Original ATen: [aten.convolution, aten.silu]
    buf67 = extern_kernels.convolution(buf66, arg135_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
    assert_size_stride(buf67, (8, 432, 14, 14), (84672, 1, 6048, 432))
    del arg135_1
    del buf66
    buf68 = buf67; del buf67  # reuse
    buf69 = empty_strided((8, 432, 1, 1), (432, 1, 3456, 3456), device='cpu', dtype=torch.float32)
    buf70 = reinterpret_tensor(buf69, (8, 432, 1, 1), (432, 1, 432, 432), 0); del buf69  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_26(c_void_p(buf68.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()))
    del arg265_1
    del arg266_1
    del arg38_1
    del arg39_1
    # Source Nodes: [x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean]
    buf71 = extern_kernels.convolution(buf70, arg136_1, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf71, (8, 36, 1, 1), (36, 1, 36, 36))
    del arg136_1
    del arg137_1
    del buf70
    buf72 = buf71; del buf71  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_27(c_void_p(buf72.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()))
    del arg138_1
    del arg139_1
    del arg334_1
    del arg335_1
    # Source Nodes: [getattr_l__mod___features___6___se_bn, x_se_14, x_se_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf73 = extern_kernels.convolution(buf72, arg140_1, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf73, (8, 432, 1, 1), (432, 1, 432, 432))
    del arg140_1
    del arg141_1
    del buf72
    buf74 = buf68; del buf68  # reuse
    cpp_fused_hardtanh_mul_sigmoid_28(c_void_p(buf74.data_ptr()), c_void_p(buf73.data_ptr()))
    del buf73
    # Source Nodes: [sigmoid_3, x_131, x_132, x_133], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf75 = extern_kernels.convolution(buf74, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf75, (8, 84, 14, 14), (16464, 1, 1176, 84))
    del arg142_1
    del buf74
    buf76 = buf75; del buf75  # reuse
    buf77 = buf76; del buf76  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_29(c_void_p(buf77.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf63.data_ptr()))
    del arg267_1
    del arg268_1
    del arg40_1
    del arg41_1
    del buf63
    # Source Nodes: [x_140], Original ATen: [aten.convolution]
    buf78 = extern_kernels.convolution(buf77, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf78, (8, 504, 14, 14), (98784, 1, 7056, 504))
    del arg143_1
    buf79 = buf78; del buf78  # reuse
    buf80 = buf79; del buf79  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_30(c_void_p(buf80.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()))
    del arg269_1
    del arg270_1
    del arg42_1
    del arg43_1
    # Source Nodes: [x_145, x_146], Original ATen: [aten.convolution, aten.silu]
    buf81 = extern_kernels.convolution(buf80, arg144_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=504, bias=None)
    assert_size_stride(buf81, (8, 504, 14, 14), (98784, 1, 7056, 504))
    del arg144_1
    del buf80
    buf82 = buf81; del buf81  # reuse
    buf83 = empty_strided((8, 504, 1, 1), (504, 1, 4032, 4032), device='cpu', dtype=torch.float32)
    buf84 = reinterpret_tensor(buf83, (8, 504, 1, 1), (504, 1, 504, 504), 0); del buf83  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_31(c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()))
    del arg271_1
    del arg272_1
    del arg44_1
    del arg45_1
    # Source Nodes: [x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean]
    buf85 = extern_kernels.convolution(buf84, arg145_1, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf85, (8, 42, 1, 1), (42, 1, 42, 42))
    del arg145_1
    del arg146_1
    del buf84
    buf86 = buf85; del buf85  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_32(c_void_p(buf86.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(arg148_1.data_ptr()))
    del arg147_1
    del arg148_1
    del arg337_1
    del arg338_1
    # Source Nodes: [getattr_l__mod___features___7___se_bn, x_se_18, x_se_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf87 = extern_kernels.convolution(buf86, arg149_1, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf87, (8, 504, 1, 1), (504, 1, 504, 504))
    del arg149_1
    del arg150_1
    del buf86
    buf88 = buf82; del buf82  # reuse
    cpp_fused_hardtanh_mul_sigmoid_33(c_void_p(buf88.data_ptr()), c_void_p(buf87.data_ptr()))
    del buf87
    # Source Nodes: [sigmoid_4, x_152, x_153, x_154], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf89 = extern_kernels.convolution(buf88, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf89, (8, 95, 14, 14), (18620, 1, 1330, 95))
    del arg151_1
    del buf88
    buf90 = buf89; del buf89  # reuse
    buf91 = buf90; del buf90  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_34(c_void_p(buf91.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf77.data_ptr()))
    del arg273_1
    del arg274_1
    del arg46_1
    del arg47_1
    del buf77
    # Source Nodes: [x_161], Original ATen: [aten.convolution]
    buf92 = extern_kernels.convolution(buf91, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf92, (8, 570, 14, 14), (111720, 1, 7980, 570))
    del arg152_1
    buf93 = buf92; del buf92  # reuse
    buf94 = buf93; del buf93  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_35(c_void_p(buf94.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()))
    del arg275_1
    del arg276_1
    del arg48_1
    del arg49_1
    # Source Nodes: [x_166, x_167], Original ATen: [aten.convolution, aten.silu]
    buf95 = extern_kernels.convolution(buf94, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=570, bias=None)
    assert_size_stride(buf95, (8, 570, 14, 14), (111720, 1, 7980, 570))
    del arg153_1
    del buf94
    buf96 = buf95; del buf95  # reuse
    buf97 = empty_strided((8, 570, 1, 1), (570, 1, 4560, 4560), device='cpu', dtype=torch.float32)
    buf98 = reinterpret_tensor(buf97, (8, 570, 1, 1), (570, 1, 570, 570), 0); del buf97  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_36(c_void_p(buf96.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()))
    del arg277_1
    del arg278_1
    del arg50_1
    del arg51_1
    # Source Nodes: [x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean]
    buf99 = extern_kernels.convolution(buf98, arg154_1, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf99, (8, 47, 1, 1), (47, 1, 47, 47))
    del arg154_1
    del arg155_1
    del buf98
    buf100 = buf99; del buf99  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_37(c_void_p(buf100.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg157_1.data_ptr()))
    del arg156_1
    del arg157_1
    del arg340_1
    del arg341_1
    # Source Nodes: [getattr_l__mod___features___8___se_bn, x_se_22, x_se_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf101 = extern_kernels.convolution(buf100, arg158_1, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf101, (8, 570, 1, 1), (570, 1, 570, 570))
    del arg158_1
    del arg159_1
    del buf100
    buf102 = buf96; del buf96  # reuse
    cpp_fused_hardtanh_mul_sigmoid_38(c_void_p(buf102.data_ptr()), c_void_p(buf101.data_ptr()))
    del buf101
    # Source Nodes: [sigmoid_5, x_173, x_174, x_175], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf103 = extern_kernels.convolution(buf102, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf103, (8, 106, 14, 14), (20776, 1, 1484, 106))
    del arg160_1
    del buf102
    buf104 = buf103; del buf103  # reuse
    buf105 = buf104; del buf104  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_39(c_void_p(buf105.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf91.data_ptr()))
    del arg279_1
    del arg280_1
    del arg52_1
    del arg53_1
    del buf91
    # Source Nodes: [x_182], Original ATen: [aten.convolution]
    buf106 = extern_kernels.convolution(buf105, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf106, (8, 636, 14, 14), (124656, 1, 8904, 636))
    del arg161_1
    buf107 = buf106; del buf106  # reuse
    buf108 = buf107; del buf107  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_40(c_void_p(buf108.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()))
    del arg281_1
    del arg282_1
    del arg54_1
    del arg55_1
    # Source Nodes: [x_187, x_188], Original ATen: [aten.convolution, aten.silu]
    buf109 = extern_kernels.convolution(buf108, arg162_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=636, bias=None)
    assert_size_stride(buf109, (8, 636, 14, 14), (124656, 1, 8904, 636))
    del arg162_1
    del buf108
    buf110 = buf109; del buf109  # reuse
    buf111 = empty_strided((8, 636, 1, 1), (636, 1, 5088, 5088), device='cpu', dtype=torch.float32)
    buf112 = reinterpret_tensor(buf111, (8, 636, 1, 1), (636, 1, 636, 636), 0); del buf111  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_41(c_void_p(buf110.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()))
    del arg283_1
    del arg284_1
    del arg56_1
    del arg57_1
    # Source Nodes: [x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean]
    buf113 = extern_kernels.convolution(buf112, arg163_1, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf113, (8, 53, 1, 1), (53, 1, 53, 53))
    del arg163_1
    del arg164_1
    del buf112
    buf114 = buf113; del buf113  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_42(c_void_p(buf114.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg166_1.data_ptr()))
    del arg165_1
    del arg166_1
    del arg343_1
    del arg344_1
    # Source Nodes: [getattr_l__mod___features___9___se_bn, x_se_26, x_se_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf115 = extern_kernels.convolution(buf114, arg167_1, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf115, (8, 636, 1, 1), (636, 1, 636, 636))
    del arg167_1
    del arg168_1
    del buf114
    buf116 = buf110; del buf110  # reuse
    cpp_fused_hardtanh_mul_sigmoid_43(c_void_p(buf116.data_ptr()), c_void_p(buf115.data_ptr()))
    del buf115
    # Source Nodes: [sigmoid_6, x_194, x_195, x_196], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf117 = extern_kernels.convolution(buf116, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf117, (8, 117, 14, 14), (22932, 1, 1638, 117))
    del arg169_1
    del buf116
    buf118 = buf117; del buf117  # reuse
    buf119 = buf118; del buf118  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_44(c_void_p(buf119.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(buf105.data_ptr()))
    del arg285_1
    del arg286_1
    del arg58_1
    del arg59_1
    del buf105
    # Source Nodes: [x_203], Original ATen: [aten.convolution]
    buf120 = extern_kernels.convolution(buf119, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf120, (8, 702, 14, 14), (137592, 1, 9828, 702))
    del arg170_1
    buf121 = buf120; del buf120  # reuse
    buf122 = buf121; del buf121  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_45(c_void_p(buf122.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()))
    del arg287_1
    del arg288_1
    del arg60_1
    del arg61_1
    # Source Nodes: [x_208, x_209], Original ATen: [aten.convolution, aten.silu]
    buf123 = extern_kernels.convolution(buf122, arg171_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=702, bias=None)
    assert_size_stride(buf123, (8, 702, 14, 14), (137592, 1, 9828, 702))
    del arg171_1
    del buf122
    buf124 = buf123; del buf123  # reuse
    buf125 = empty_strided((8, 702, 1, 1), (702, 1, 5616, 5616), device='cpu', dtype=torch.float32)
    buf126 = reinterpret_tensor(buf125, (8, 702, 1, 1), (702, 1, 702, 702), 0); del buf125  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_46(c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()))
    del arg289_1
    del arg290_1
    del arg62_1
    del arg63_1
    # Source Nodes: [x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean]
    buf127 = extern_kernels.convolution(buf126, arg172_1, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf127, (8, 58, 1, 1), (58, 1, 58, 58))
    del arg172_1
    del arg173_1
    del buf126
    buf128 = buf127; del buf127  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_47(c_void_p(buf128.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()))
    del arg174_1
    del arg175_1
    del arg346_1
    del arg347_1
    # Source Nodes: [getattr_l__mod___features___10___se_bn, x_se_30, x_se_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf129 = extern_kernels.convolution(buf128, arg176_1, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf129, (8, 702, 1, 1), (702, 1, 702, 702))
    del arg176_1
    del arg177_1
    del buf128
    buf130 = buf124; del buf124  # reuse
    cpp_fused_hardtanh_mul_sigmoid_48(c_void_p(buf130.data_ptr()), c_void_p(buf129.data_ptr()))
    del buf129
    # Source Nodes: [sigmoid_7, x_215, x_216, x_217], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf131 = extern_kernels.convolution(buf130, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf131, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del arg178_1
    del buf130
    buf132 = buf131; del buf131  # reuse
    buf133 = buf132; del buf132  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_49(c_void_p(buf133.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(buf119.data_ptr()))
    del arg291_1
    del arg292_1
    del arg64_1
    del arg65_1
    del buf119
    # Source Nodes: [cat_15, x_224], Original ATen: [aten.cat, aten.convolution]
    buf134 = extern_kernels.convolution(buf133, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf134, (8, 768, 14, 14), (150528, 1, 10752, 768))
    del arg179_1
    del buf133
    buf135 = buf134; del buf134  # reuse
    buf136 = buf135; del buf135  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_50(c_void_p(buf136.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg293_1
    del arg294_1
    del arg66_1
    del arg67_1
    # Source Nodes: [x_229, x_230], Original ATen: [aten.convolution, aten.silu]
    buf137 = extern_kernels.convolution(buf136, arg180_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
    assert_size_stride(buf137, (8, 768, 7, 7), (37632, 1, 5376, 768))
    del arg180_1
    del buf136
    buf138 = buf137; del buf137  # reuse
    buf139 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cpu', dtype=torch.float32)
    buf140 = reinterpret_tensor(buf139, (8, 768, 1, 1), (768, 1, 768, 768), 0); del buf139  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_51(c_void_p(buf138.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()))
    del arg295_1
    del arg296_1
    del arg68_1
    del arg69_1
    # Source Nodes: [x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean]
    buf141 = extern_kernels.convolution(buf140, arg181_1, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf141, (8, 64, 1, 1), (64, 1, 64, 64))
    del arg181_1
    del arg182_1
    del buf140
    buf142 = buf141; del buf141  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_52(c_void_p(buf142.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(arg184_1.data_ptr()))
    del arg183_1
    del arg184_1
    del arg349_1
    del arg350_1
    # Source Nodes: [getattr_l__mod___features___11___se_bn, x_se_34, x_se_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf143 = extern_kernels.convolution(buf142, arg185_1, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf143, (8, 768, 1, 1), (768, 1, 768, 768))
    del arg185_1
    del arg186_1
    del buf142
    buf144 = buf138; del buf138  # reuse
    cpp_fused_hardtanh_mul_sigmoid_53(c_void_p(buf144.data_ptr()), c_void_p(buf143.data_ptr()))
    del buf143
    # Source Nodes: [sigmoid_8, x_236, x_237, x_238], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf145 = extern_kernels.convolution(buf144, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf145, (8, 140, 7, 7), (6860, 1, 980, 140))
    del arg187_1
    del buf144
    buf146 = buf145; del buf145  # reuse
    cpp_fused__native_batch_norm_legit_no_training_54(c_void_p(buf146.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg297_1
    del arg298_1
    del arg70_1
    del arg71_1
    # Source Nodes: [x_244], Original ATen: [aten.convolution]
    buf147 = extern_kernels.convolution(buf146, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf147, (8, 840, 7, 7), (41160, 1, 5880, 840))
    del arg188_1
    buf148 = buf147; del buf147  # reuse
    buf149 = buf148; del buf148  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_55(c_void_p(buf149.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()))
    del arg299_1
    del arg300_1
    del arg72_1
    del arg73_1
    # Source Nodes: [x_249, x_250], Original ATen: [aten.convolution, aten.silu]
    buf150 = extern_kernels.convolution(buf149, arg189_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=840, bias=None)
    assert_size_stride(buf150, (8, 840, 7, 7), (41160, 1, 5880, 840))
    del arg189_1
    del buf149
    buf151 = buf150; del buf150  # reuse
    buf152 = empty_strided((8, 840, 1, 1), (840, 1, 6720, 6720), device='cpu', dtype=torch.float32)
    buf153 = reinterpret_tensor(buf152, (8, 840, 1, 1), (840, 1, 840, 840), 0); del buf152  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_56(c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()))
    del arg301_1
    del arg302_1
    del arg74_1
    del arg75_1
    # Source Nodes: [x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean]
    buf154 = extern_kernels.convolution(buf153, arg190_1, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf154, (8, 70, 1, 1), (70, 1, 70, 70))
    del arg190_1
    del arg191_1
    del buf153
    buf155 = buf154; del buf154  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_57(c_void_p(buf155.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()))
    del arg192_1
    del arg193_1
    del arg352_1
    del arg353_1
    # Source Nodes: [getattr_l__mod___features___12___se_bn, x_se_38, x_se_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf156 = extern_kernels.convolution(buf155, arg194_1, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf156, (8, 840, 1, 1), (840, 1, 840, 840))
    del arg194_1
    del arg195_1
    del buf155
    buf157 = buf151; del buf151  # reuse
    cpp_fused_hardtanh_mul_sigmoid_58(c_void_p(buf157.data_ptr()), c_void_p(buf156.data_ptr()))
    del buf156
    # Source Nodes: [sigmoid_9, x_256, x_257, x_258], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf158 = extern_kernels.convolution(buf157, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf158, (8, 151, 7, 7), (7399, 1, 1057, 151))
    del arg196_1
    del buf157
    buf159 = buf158; del buf158  # reuse
    buf160 = buf159; del buf159  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_59(c_void_p(buf160.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf146.data_ptr()))
    del arg303_1
    del arg304_1
    del arg76_1
    del arg77_1
    del buf146
    # Source Nodes: [x_265], Original ATen: [aten.convolution]
    buf161 = extern_kernels.convolution(buf160, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf161, (8, 906, 7, 7), (44394, 1, 6342, 906))
    del arg197_1
    buf162 = buf161; del buf161  # reuse
    buf163 = buf162; del buf162  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_60(c_void_p(buf163.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()))
    del arg305_1
    del arg306_1
    del arg78_1
    del arg79_1
    # Source Nodes: [x_270, x_271], Original ATen: [aten.convolution, aten.silu]
    buf164 = extern_kernels.convolution(buf163, arg198_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=906, bias=None)
    assert_size_stride(buf164, (8, 906, 7, 7), (44394, 1, 6342, 906))
    del arg198_1
    del buf163
    buf165 = buf164; del buf164  # reuse
    buf166 = empty_strided((8, 906, 1, 1), (906, 1, 7248, 7248), device='cpu', dtype=torch.float32)
    buf167 = reinterpret_tensor(buf166, (8, 906, 1, 1), (906, 1, 906, 906), 0); del buf166  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_61(c_void_p(buf165.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()))
    del arg307_1
    del arg308_1
    del arg80_1
    del arg81_1
    # Source Nodes: [x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean]
    buf168 = extern_kernels.convolution(buf167, arg199_1, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf168, (8, 75, 1, 1), (75, 1, 75, 75))
    del arg199_1
    del arg200_1
    del buf167
    buf169 = buf168; del buf168  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_62(c_void_p(buf169.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg202_1.data_ptr()))
    del arg201_1
    del arg202_1
    del arg355_1
    del arg356_1
    # Source Nodes: [getattr_l__mod___features___13___se_bn, x_se_42, x_se_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf170 = extern_kernels.convolution(buf169, arg203_1, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf170, (8, 906, 1, 1), (906, 1, 906, 906))
    del arg203_1
    del arg204_1
    del buf169
    buf171 = buf165; del buf165  # reuse
    cpp_fused_hardtanh_mul_sigmoid_63(c_void_p(buf171.data_ptr()), c_void_p(buf170.data_ptr()))
    del buf170
    # Source Nodes: [sigmoid_10, x_277, x_278, x_279], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf172 = extern_kernels.convolution(buf171, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf172, (8, 162, 7, 7), (7938, 1, 1134, 162))
    del arg205_1
    del buf171
    buf173 = buf172; del buf172  # reuse
    buf174 = buf173; del buf173  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_64(c_void_p(buf174.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(buf160.data_ptr()))
    del arg309_1
    del arg310_1
    del arg82_1
    del arg83_1
    del buf160
    # Source Nodes: [x_286], Original ATen: [aten.convolution]
    buf175 = extern_kernels.convolution(buf174, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf175, (8, 972, 7, 7), (47628, 1, 6804, 972))
    del arg206_1
    buf176 = buf175; del buf175  # reuse
    buf177 = buf176; del buf176  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_65(c_void_p(buf177.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()))
    del arg311_1
    del arg312_1
    del arg84_1
    del arg85_1
    # Source Nodes: [x_291, x_292], Original ATen: [aten.convolution, aten.silu]
    buf178 = extern_kernels.convolution(buf177, arg207_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=972, bias=None)
    assert_size_stride(buf178, (8, 972, 7, 7), (47628, 1, 6804, 972))
    del arg207_1
    del buf177
    buf179 = buf178; del buf178  # reuse
    buf180 = empty_strided((8, 972, 1, 1), (972, 1, 7776, 7776), device='cpu', dtype=torch.float32)
    buf181 = reinterpret_tensor(buf180, (8, 972, 1, 1), (972, 1, 972, 972), 0); del buf180  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_66(c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()))
    del arg313_1
    del arg314_1
    del arg86_1
    del arg87_1
    # Source Nodes: [x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean]
    buf182 = extern_kernels.convolution(buf181, arg208_1, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf182, (8, 81, 1, 1), (81, 1, 81, 81))
    del arg208_1
    del arg209_1
    del buf181
    buf183 = buf182; del buf182  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_67(c_void_p(buf183.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()))
    del arg210_1
    del arg211_1
    del arg358_1
    del arg359_1
    # Source Nodes: [getattr_l__mod___features___14___se_bn, x_se_46, x_se_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf184 = extern_kernels.convolution(buf183, arg212_1, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf184, (8, 972, 1, 1), (972, 1, 972, 972))
    del arg212_1
    del arg213_1
    del buf183
    buf185 = buf179; del buf179  # reuse
    cpp_fused_hardtanh_mul_sigmoid_68(c_void_p(buf185.data_ptr()), c_void_p(buf184.data_ptr()))
    del buf184
    # Source Nodes: [sigmoid_11, x_298, x_299, x_300], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf186 = extern_kernels.convolution(buf185, arg214_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf186, (8, 174, 7, 7), (8526, 1, 1218, 174))
    del arg214_1
    del buf185
    buf187 = buf186; del buf186  # reuse
    buf188 = buf187; del buf187  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_69(c_void_p(buf188.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(buf174.data_ptr()))
    del arg315_1
    del arg316_1
    del arg88_1
    del arg89_1
    del buf174
    # Source Nodes: [x_307], Original ATen: [aten.convolution]
    buf189 = extern_kernels.convolution(buf188, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf189, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    del arg215_1
    buf190 = buf189; del buf189  # reuse
    buf191 = buf190; del buf190  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_70(c_void_p(buf191.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()))
    del arg317_1
    del arg318_1
    del arg90_1
    del arg91_1
    # Source Nodes: [x_312, x_313], Original ATen: [aten.convolution, aten.silu]
    buf192 = extern_kernels.convolution(buf191, arg216_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1044, bias=None)
    assert_size_stride(buf192, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    del arg216_1
    del buf191
    buf193 = buf192; del buf192  # reuse
    buf194 = empty_strided((8, 1044, 1, 1), (1044, 1, 8352, 8352), device='cpu', dtype=torch.float32)
    buf195 = reinterpret_tensor(buf194, (8, 1044, 1, 1), (1044, 1, 1044, 1044), 0); del buf194  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_71(c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()))
    del arg319_1
    del arg320_1
    del arg92_1
    del arg93_1
    # Source Nodes: [x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean]
    buf196 = extern_kernels.convolution(buf195, arg217_1, arg218_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf196, (8, 87, 1, 1), (87, 1, 87, 87))
    del arg217_1
    del arg218_1
    del buf195
    buf197 = buf196; del buf196  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_72(c_void_p(buf197.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg220_1.data_ptr()))
    del arg219_1
    del arg220_1
    del arg361_1
    del arg362_1
    # Source Nodes: [getattr_l__mod___features___15___se_bn, x_se_50, x_se_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf198 = extern_kernels.convolution(buf197, arg221_1, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf198, (8, 1044, 1, 1), (1044, 1, 1044, 1044))
    del arg221_1
    del arg222_1
    del buf197
    buf199 = buf193; del buf193  # reuse
    cpp_fused_hardtanh_mul_sigmoid_73(c_void_p(buf199.data_ptr()), c_void_p(buf198.data_ptr()))
    del buf198
    # Source Nodes: [sigmoid_12, x_319, x_320, x_321], Original ATen: [aten.convolution, aten.hardtanh, aten.mul, aten.sigmoid]
    buf200 = extern_kernels.convolution(buf199, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf200, (8, 185, 7, 7), (9065, 1, 1295, 185))
    del arg223_1
    del buf199
    buf201 = buf200; del buf200  # reuse
    buf202 = buf201; del buf201  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_74(c_void_p(buf202.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(buf188.data_ptr()))
    del arg321_1
    del arg322_1
    del arg94_1
    del arg95_1
    del buf188
    # Source Nodes: [cat_11, x_328], Original ATen: [aten.cat, aten.convolution]
    buf203 = extern_kernels.convolution(buf202, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf203, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    del arg224_1
    del buf202
    buf204 = buf203; del buf203  # reuse
    buf205 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cpu', dtype=torch.float32)
    buf206 = reinterpret_tensor(buf205, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf205  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_75(c_void_p(buf204.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()))
    del arg323_1
    del arg324_1
    del arg96_1
    del arg97_1
    del buf204
    buf207 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_339], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg226_1, reinterpret_tensor(buf206, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg225_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf207)
    del arg225_1
    del arg226_1
    return (buf207, )


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
    arg10_1 = rand_strided((27, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((27, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((50, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((50, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((61, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((61, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((84, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((84, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((95, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((95, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((106, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((106, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((117, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((117, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((140, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((140, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((151, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((151, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((174, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((174, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((185, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((185, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((27, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((162, 27, 1, 1), (27, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((162, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((38, 162, 1, 1), (162, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((228, 38, 1, 1), (38, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((228, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((19, 228, 1, 1), (228, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((19, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((19, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((19, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((228, 19, 1, 1), (19, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((50, 228, 1, 1), (228, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((300, 50, 1, 1), (50, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((300, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((25, 300, 1, 1), (300, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((25, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((25, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((25, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((300, 25, 1, 1), (25, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((61, 300, 1, 1), (300, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((366, 61, 1, 1), (61, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((366, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((30, 366, 1, 1), (366, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((30, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((30, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((30, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((366, 30, 1, 1), (30, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((72, 366, 1, 1), (366, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((432, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((36, 432, 1, 1), (432, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((432, 36, 1, 1), (36, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((84, 432, 1, 1), (432, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((504, 84, 1, 1), (84, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((504, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((42, 504, 1, 1), (504, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((42, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((42, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((42, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((504, 42, 1, 1), (42, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((95, 504, 1, 1), (504, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((570, 95, 1, 1), (95, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((570, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((47, 570, 1, 1), (570, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((47, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((47, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((47, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((570, 47, 1, 1), (47, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((106, 570, 1, 1), (570, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((636, 106, 1, 1), (106, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((636, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((53, 636, 1, 1), (636, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((53, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((53, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((53, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((636, 53, 1, 1), (53, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((117, 636, 1, 1), (636, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((702, 117, 1, 1), (117, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((702, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((58, 702, 1, 1), (702, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((702, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((128, 702, 1, 1), (702, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((768, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((64, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((768, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((140, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((840, 140, 1, 1), (140, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((840, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((70, 840, 1, 1), (840, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((70, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((70, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((70, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((840, 70, 1, 1), (70, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((151, 840, 1, 1), (840, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((906, 151, 1, 1), (151, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((906, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((75, 906, 1, 1), (906, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((75, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((75, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((75, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((906, 75, 1, 1), (75, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((162, 906, 1, 1), (906, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((972, 162, 1, 1), (162, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((972, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((81, 972, 1, 1), (972, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((81, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((81, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((81, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((972, 81, 1, 1), (81, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((174, 972, 1, 1), (972, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((1044, 174, 1, 1), (174, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((1044, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((87, 1044, 1, 1), (1044, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((87, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((87, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((87, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((1044, 87, 1, 1), (87, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((185, 1044, 1, 1), (1044, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((1280, 185, 1, 1), (185, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((27, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((27, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((50, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((50, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((61, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((61, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((84, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((84, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((95, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((95, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((106, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((106, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((117, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((117, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((140, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((140, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((151, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((151, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((174, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((174, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((185, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((185, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((19, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((19, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg328_1 = rand_strided((25, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((25, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg331_1 = rand_strided((30, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((30, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg334_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg337_1 = rand_strided((42, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((42, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg340_1 = rand_strided((47, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((47, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg343_1 = rand_strided((53, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((53, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg346_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg349_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg352_1 = rand_strided((70, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((70, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg355_1 = rand_strided((75, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((75, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg358_1 = rand_strided((81, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((81, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg361_1 = rand_strided((87, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((87, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg364_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('rexnet_100', benchmark_compiled_module)
