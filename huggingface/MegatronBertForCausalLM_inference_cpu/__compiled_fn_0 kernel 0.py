
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


cpp_fused_add_embedding_native_layer_norm_zeros_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const long* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 29056);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 29056L), "index out of bounds: 0 <= tmp3 < 29056L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*tmp3)));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = decltype(tmp7)(tmp7 + 512);
                        auto tmp9 = tmp7 < 0;
                        auto tmp10 = tmp9 ? tmp8 : tmp7;
                        TORCH_CHECK((0 <= tmp10) & (tmp10 < 512L), "index out of bounds: 0 <= tmp10 < 512L")
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*tmp10)));
                        auto tmp12 = tmp6 + tmp11;
                        tmp12.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp12);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-12);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_8 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-12);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_16 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-12);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_24 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-12);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_32 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-12);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_40 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-12);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_48 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-12);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_56 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-12);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_64 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-12);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_72 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-12);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_80 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-12);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_88 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-12);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                tmp0.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_96 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_native_layer_norm_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = out_ptr1[static_cast<long>(x0)];
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.5);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp16 = static_cast<float>(1024.0);
                    auto tmp17 = tmp15 / tmp16;
                    auto tmp18 = static_cast<float>(1e-12);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = 1 / std::sqrt(tmp19);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp14 * tmp21;
                    auto tmp24 = tmp22 * tmp23;
                    auto tmp26 = tmp24 + tmp25;
                    tmp26.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_nll_loss_forward_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(511L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(29056L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (29056L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(511L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(29056L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (29056L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    float tmp_acc0 = 0;
                    long tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(511L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(1L + x0)];
                        auto tmp9 = out_ptr0[static_cast<long>(x0)];
                        auto tmp11 = out_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = static_cast<long>(0);
                        auto tmp4 = tmp2 ? tmp0 : tmp3;
                        auto tmp5 = decltype(tmp4)(tmp4 + 29056);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 29056L), "index out of bounds: 0 <= tmp7 < 29056L")
                        auto tmp8 = in_ptr0[static_cast<long>(tmp7 + (29056L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                        auto tmp12 = std::log(tmp11);
                        auto tmp13 = decltype(tmp10)(tmp10 - tmp12);
                        auto tmp14 = decltype(tmp13)(-tmp13);
                        auto tmp15 = static_cast<float>(0.0);
                        auto tmp16 = tmp2 ? tmp14 : tmp15;
                        auto tmp17 = c10::convert<long>(tmp2);
                        tmp_acc0 = tmp_acc0 + tmp16;
                        tmp_acc1 = tmp_acc1 + tmp17;
                    }
                    out_ptr2[static_cast<long>(0L)] = tmp_acc0;
                    out_ptr3[static_cast<long>(0L)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = out_ptr2[static_cast<long>(0L)];
                auto tmp1 = out_ptr3[static_cast<long>(0L)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                in_out_ptr0[static_cast<long>(0L)] = tmp3;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1 = args
    args.clear()
    assert_size_stride(arg0_1, (29056, 1024), (1024, 1))
    assert_size_stride(arg1_1, (2, 1024), (1024, 1))
    assert_size_stride(arg2_1, (512, 1024), (1024, 1))
    assert_size_stride(arg3_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (1024, ), (1, ))
    assert_size_stride(arg5_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg6_1, (1024, ), (1, ))
    assert_size_stride(arg7_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg8_1, (1024, ), (1, ))
    assert_size_stride(arg9_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg10_1, (1024, ), (1, ))
    assert_size_stride(arg11_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg12_1, (1024, ), (1, ))
    assert_size_stride(arg13_1, (1024, ), (1, ))
    assert_size_stride(arg14_1, (1024, ), (1, ))
    assert_size_stride(arg15_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg16_1, (4096, ), (1, ))
    assert_size_stride(arg17_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg18_1, (1024, ), (1, ))
    assert_size_stride(arg19_1, (1024, ), (1, ))
    assert_size_stride(arg20_1, (1024, ), (1, ))
    assert_size_stride(arg21_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg22_1, (1024, ), (1, ))
    assert_size_stride(arg23_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg24_1, (1024, ), (1, ))
    assert_size_stride(arg25_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg26_1, (1024, ), (1, ))
    assert_size_stride(arg27_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg28_1, (1024, ), (1, ))
    assert_size_stride(arg29_1, (1024, ), (1, ))
    assert_size_stride(arg30_1, (1024, ), (1, ))
    assert_size_stride(arg31_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg32_1, (4096, ), (1, ))
    assert_size_stride(arg33_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg34_1, (1024, ), (1, ))
    assert_size_stride(arg35_1, (1024, ), (1, ))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg40_1, (1024, ), (1, ))
    assert_size_stride(arg41_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg42_1, (1024, ), (1, ))
    assert_size_stride(arg43_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (1024, ), (1, ))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg48_1, (4096, ), (1, ))
    assert_size_stride(arg49_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (1024, ), (1, ))
    assert_size_stride(arg52_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg57_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (1024, ), (1, ))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg63_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg64_1, (4096, ), (1, ))
    assert_size_stride(arg65_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg70_1, (1024, ), (1, ))
    assert_size_stride(arg71_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg72_1, (1024, ), (1, ))
    assert_size_stride(arg73_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg74_1, (1024, ), (1, ))
    assert_size_stride(arg75_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (1024, ), (1, ))
    assert_size_stride(arg79_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg80_1, (4096, ), (1, ))
    assert_size_stride(arg81_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg87_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg88_1, (1024, ), (1, ))
    assert_size_stride(arg89_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg90_1, (1024, ), (1, ))
    assert_size_stride(arg91_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (1024, ), (1, ))
    assert_size_stride(arg94_1, (1024, ), (1, ))
    assert_size_stride(arg95_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg96_1, (4096, ), (1, ))
    assert_size_stride(arg97_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg98_1, (1024, ), (1, ))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg102_1, (1024, ), (1, ))
    assert_size_stride(arg103_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg104_1, (1024, ), (1, ))
    assert_size_stride(arg105_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg106_1, (1024, ), (1, ))
    assert_size_stride(arg107_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg112_1, (4096, ), (1, ))
    assert_size_stride(arg113_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg120_1, (1024, ), (1, ))
    assert_size_stride(arg121_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg122_1, (1024, ), (1, ))
    assert_size_stride(arg123_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg124_1, (1024, ), (1, ))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (1024, ), (1, ))
    assert_size_stride(arg127_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg128_1, (4096, ), (1, ))
    assert_size_stride(arg129_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (1024, ), (1, ))
    assert_size_stride(arg142_1, (1024, ), (1, ))
    assert_size_stride(arg143_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg144_1, (4096, ), (1, ))
    assert_size_stride(arg145_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (1024, ), (1, ))
    assert_size_stride(arg158_1, (1024, ), (1, ))
    assert_size_stride(arg159_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg160_1, (4096, ), (1, ))
    assert_size_stride(arg161_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (1024, ), (1, ))
    assert_size_stride(arg174_1, (1024, ), (1, ))
    assert_size_stride(arg175_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg176_1, (4096, ), (1, ))
    assert_size_stride(arg177_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (1024, ), (1, ))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg186_1, (1024, ), (1, ))
    assert_size_stride(arg187_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg188_1, (1024, ), (1, ))
    assert_size_stride(arg189_1, (1024, ), (1, ))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg192_1, (4096, ), (1, ))
    assert_size_stride(arg193_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (1024, ), (1, ))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg202_1, (1024, ), (1, ))
    assert_size_stride(arg203_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg204_1, (1024, ), (1, ))
    assert_size_stride(arg205_1, (1024, ), (1, ))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg208_1, (4096, ), (1, ))
    assert_size_stride(arg209_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg210_1, (1024, ), (1, ))
    assert_size_stride(arg211_1, (1024, ), (1, ))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg216_1, (1024, ), (1, ))
    assert_size_stride(arg217_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg218_1, (1024, ), (1, ))
    assert_size_stride(arg219_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg220_1, (1024, ), (1, ))
    assert_size_stride(arg221_1, (1024, ), (1, ))
    assert_size_stride(arg222_1, (1024, ), (1, ))
    assert_size_stride(arg223_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg224_1, (4096, ), (1, ))
    assert_size_stride(arg225_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (1024, ), (1, ))
    assert_size_stride(arg229_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg230_1, (1024, ), (1, ))
    assert_size_stride(arg231_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg232_1, (1024, ), (1, ))
    assert_size_stride(arg233_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg234_1, (1024, ), (1, ))
    assert_size_stride(arg235_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg236_1, (1024, ), (1, ))
    assert_size_stride(arg237_1, (1024, ), (1, ))
    assert_size_stride(arg238_1, (1024, ), (1, ))
    assert_size_stride(arg239_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg240_1, (4096, ), (1, ))
    assert_size_stride(arg241_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (1024, ), (1, ))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg246_1, (1024, ), (1, ))
    assert_size_stride(arg247_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg248_1, (1024, ), (1, ))
    assert_size_stride(arg249_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg250_1, (1024, ), (1, ))
    assert_size_stride(arg251_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg252_1, (1024, ), (1, ))
    assert_size_stride(arg253_1, (1024, ), (1, ))
    assert_size_stride(arg254_1, (1024, ), (1, ))
    assert_size_stride(arg255_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg256_1, (4096, ), (1, ))
    assert_size_stride(arg257_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg258_1, (1024, ), (1, ))
    assert_size_stride(arg259_1, (1024, ), (1, ))
    assert_size_stride(arg260_1, (1024, ), (1, ))
    assert_size_stride(arg261_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg262_1, (1024, ), (1, ))
    assert_size_stride(arg263_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg264_1, (1024, ), (1, ))
    assert_size_stride(arg265_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg266_1, (1024, ), (1, ))
    assert_size_stride(arg267_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg268_1, (1024, ), (1, ))
    assert_size_stride(arg269_1, (1024, ), (1, ))
    assert_size_stride(arg270_1, (1024, ), (1, ))
    assert_size_stride(arg271_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg272_1, (4096, ), (1, ))
    assert_size_stride(arg273_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg274_1, (1024, ), (1, ))
    assert_size_stride(arg275_1, (1024, ), (1, ))
    assert_size_stride(arg276_1, (1024, ), (1, ))
    assert_size_stride(arg277_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg278_1, (1024, ), (1, ))
    assert_size_stride(arg279_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg282_1, (1024, ), (1, ))
    assert_size_stride(arg283_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg284_1, (1024, ), (1, ))
    assert_size_stride(arg285_1, (1024, ), (1, ))
    assert_size_stride(arg286_1, (1024, ), (1, ))
    assert_size_stride(arg287_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg288_1, (4096, ), (1, ))
    assert_size_stride(arg289_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg290_1, (1024, ), (1, ))
    assert_size_stride(arg291_1, (1024, ), (1, ))
    assert_size_stride(arg292_1, (1024, ), (1, ))
    assert_size_stride(arg293_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg294_1, (1024, ), (1, ))
    assert_size_stride(arg295_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg296_1, (1024, ), (1, ))
    assert_size_stride(arg297_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg298_1, (1024, ), (1, ))
    assert_size_stride(arg299_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg300_1, (1024, ), (1, ))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg304_1, (4096, ), (1, ))
    assert_size_stride(arg305_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg306_1, (1024, ), (1, ))
    assert_size_stride(arg307_1, (1024, ), (1, ))
    assert_size_stride(arg308_1, (1024, ), (1, ))
    assert_size_stride(arg309_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg310_1, (1024, ), (1, ))
    assert_size_stride(arg311_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg312_1, (1024, ), (1, ))
    assert_size_stride(arg313_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg314_1, (1024, ), (1, ))
    assert_size_stride(arg315_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (1024, ), (1, ))
    assert_size_stride(arg319_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg320_1, (4096, ), (1, ))
    assert_size_stride(arg321_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg322_1, (1024, ), (1, ))
    assert_size_stride(arg323_1, (1024, ), (1, ))
    assert_size_stride(arg324_1, (1024, ), (1, ))
    assert_size_stride(arg325_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg328_1, (1024, ), (1, ))
    assert_size_stride(arg329_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg330_1, (1024, ), (1, ))
    assert_size_stride(arg331_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg332_1, (1024, ), (1, ))
    assert_size_stride(arg333_1, (1024, ), (1, ))
    assert_size_stride(arg334_1, (1024, ), (1, ))
    assert_size_stride(arg335_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg336_1, (4096, ), (1, ))
    assert_size_stride(arg337_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg338_1, (1024, ), (1, ))
    assert_size_stride(arg339_1, (1024, ), (1, ))
    assert_size_stride(arg340_1, (1024, ), (1, ))
    assert_size_stride(arg341_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg344_1, (1024, ), (1, ))
    assert_size_stride(arg345_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg348_1, (1024, ), (1, ))
    assert_size_stride(arg349_1, (1024, ), (1, ))
    assert_size_stride(arg350_1, (1024, ), (1, ))
    assert_size_stride(arg351_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg352_1, (4096, ), (1, ))
    assert_size_stride(arg353_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg354_1, (1024, ), (1, ))
    assert_size_stride(arg355_1, (1024, ), (1, ))
    assert_size_stride(arg356_1, (1024, ), (1, ))
    assert_size_stride(arg357_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg358_1, (1024, ), (1, ))
    assert_size_stride(arg359_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg364_1, (1024, ), (1, ))
    assert_size_stride(arg365_1, (1024, ), (1, ))
    assert_size_stride(arg366_1, (1024, ), (1, ))
    assert_size_stride(arg367_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg368_1, (4096, ), (1, ))
    assert_size_stride(arg369_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg370_1, (1024, ), (1, ))
    assert_size_stride(arg371_1, (1024, ), (1, ))
    assert_size_stride(arg372_1, (1024, ), (1, ))
    assert_size_stride(arg373_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg374_1, (1024, ), (1, ))
    assert_size_stride(arg375_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg376_1, (1024, ), (1, ))
    assert_size_stride(arg377_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg380_1, (1024, ), (1, ))
    assert_size_stride(arg381_1, (1024, ), (1, ))
    assert_size_stride(arg382_1, (1024, ), (1, ))
    assert_size_stride(arg383_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg384_1, (4096, ), (1, ))
    assert_size_stride(arg385_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg386_1, (1024, ), (1, ))
    assert_size_stride(arg387_1, (1024, ), (1, ))
    assert_size_stride(arg388_1, (1024, ), (1, ))
    assert_size_stride(arg389_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg390_1, (1024, ), (1, ))
    assert_size_stride(arg391_1, (1024, ), (1, ))
    assert_size_stride(arg392_1, (1024, ), (1, ))
    assert_size_stride(arg393_1, (29056, 1024), (1024, 1))
    assert_size_stride(arg394_1, (29056, ), (1, ))
    assert_size_stride(arg395_1, (1, 512), (512, 1))
    assert_size_stride(arg396_1, (1, 512), (512, 1))
    assert_size_stride(arg397_1, (1, 512), (512, 1))
    buf0 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf4 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_zeros_0(c_void_p(arg397_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg0_1
    del arg1_1
    del arg2_1
    del arg395_1
    del arg397_1
    del arg3_1
    del arg4_1
    buf5 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg6_1, reinterpret_tensor(buf4, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg5_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf5)
    del arg5_1
    del arg6_1
    buf6 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___bert_encoder_layer_0_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf4, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg7_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf6)
    del arg7_1
    del arg8_1
    buf7 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___bert_encoder_layer_0_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf4, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg9_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf7)
    del arg10_1
    del arg9_1
    buf8 = reinterpret_tensor(buf5, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf5  # reuse
    buf9 = reinterpret_tensor(buf6, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf6  # reuse
    buf10 = reinterpret_tensor(buf7, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf7  # reuse
    cpp_fused_1(c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf11 = aten._scaled_dot_product_flash_attention(buf8, buf9, buf10, scale=0.125)
    buf12 = buf11[0]
    del buf11
    buf19 = reinterpret_tensor(buf9, (512, 1024), (1024, 1), 0); del buf9  # reuse
    # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf12, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg11_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf19)
    del arg11_1
    del arg12_1
    buf20 = buf2; del buf2  # reuse
    buf21 = buf1; del buf1  # reuse
    buf23 = reinterpret_tensor(buf12, (1, 512, 1024), (524288, 1024, 1), 0); del buf12  # reuse
    cpp_fused_add_native_layer_norm_2(c_void_p(buf0.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()))
    del arg13_1
    del arg14_1
    buf24 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg16_1, reinterpret_tensor(buf23, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg15_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf24)
    del arg15_1
    del arg16_1
    buf25 = reinterpret_tensor(buf24, (1, 512, 4096), (2097152, 4096, 1), 0); del buf24  # reuse
    cpp_fused_gelu_3(c_void_p(buf25.data_ptr()))
    buf26 = reinterpret_tensor(buf23, (512, 1024), (1024, 1), 0); del buf23  # reuse
    # Source Nodes: [hidden_states_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf25, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf26)
    del arg17_1
    del arg18_1
    buf27 = buf21; del buf21  # reuse
    buf28 = buf20; del buf20  # reuse
    buf30 = reinterpret_tensor(buf8, (1, 512, 1024), (524288, 1024, 1), 0); del buf8  # reuse
    cpp_fused_add_native_layer_norm_4(c_void_p(buf0.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()))
    del arg19_1
    del arg20_1
    buf31 = reinterpret_tensor(buf10, (512, 1024), (1024, 1), 0); del buf10  # reuse
    # Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg22_1, reinterpret_tensor(buf30, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg21_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf31)
    del arg21_1
    del arg22_1
    buf32 = reinterpret_tensor(buf4, (512, 1024), (1024, 1), 0); del buf4  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_1_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg24_1, reinterpret_tensor(buf30, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg23_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf32)
    del arg23_1
    del arg24_1
    buf33 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___bert_encoder_layer_1_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg26_1, reinterpret_tensor(buf30, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg25_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf33)
    del arg25_1
    del arg26_1
    del buf30
    buf34 = reinterpret_tensor(buf31, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf31  # reuse
    buf35 = reinterpret_tensor(buf32, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf32  # reuse
    buf36 = reinterpret_tensor(buf33, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf33  # reuse
    cpp_fused_5(c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf37 = aten._scaled_dot_product_flash_attention(buf34, buf35, buf36, scale=0.125)
    del buf34
    buf38 = buf37[0]
    del buf37
    buf45 = reinterpret_tensor(buf36, (512, 1024), (1024, 1), 0); del buf36  # reuse
    # Source Nodes: [hidden_states_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg28_1, reinterpret_tensor(buf38, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg27_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf45)
    del arg27_1
    del arg28_1
    buf46 = buf28; del buf28  # reuse
    buf47 = buf27; del buf27  # reuse
    buf49 = reinterpret_tensor(buf38, (1, 512, 1024), (524288, 1024, 1), 0); del buf38  # reuse
    cpp_fused_add_native_layer_norm_6(c_void_p(buf0.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()))
    del arg29_1
    del arg30_1
    buf50 = reinterpret_tensor(buf25, (512, 4096), (4096, 1), 0); del buf25  # reuse
    # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg32_1, reinterpret_tensor(buf49, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg31_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf50)
    del arg31_1
    del arg32_1
    buf51 = reinterpret_tensor(buf50, (1, 512, 4096), (2097152, 4096, 1), 0); del buf50  # reuse
    cpp_fused_gelu_7(c_void_p(buf51.data_ptr()))
    buf52 = reinterpret_tensor(buf49, (512, 1024), (1024, 1), 0); del buf49  # reuse
    # Source Nodes: [hidden_states_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg34_1, reinterpret_tensor(buf51, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg33_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf52)
    del arg33_1
    del arg34_1
    buf53 = reinterpret_tensor(buf52, (1, 512, 1024), (524288, 1024, 1), 0); del buf52  # reuse
    buf54 = buf47; del buf47  # reuse
    buf55 = buf46; del buf46  # reuse
    buf57 = reinterpret_tensor(buf35, (1, 512, 1024), (524288, 1024, 1), 0); del buf35  # reuse
    cpp_fused_add_native_layer_norm_8(c_void_p(buf53.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg35_1
    del arg36_1
    buf58 = buf45; del buf45  # reuse
    # Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg38_1, reinterpret_tensor(buf57, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg37_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf58)
    del arg37_1
    del arg38_1
    buf59 = buf26; del buf26  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_2_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg40_1, reinterpret_tensor(buf57, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg39_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf59)
    del arg39_1
    del arg40_1
    buf60 = buf19; del buf19  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_2_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg42_1, reinterpret_tensor(buf57, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg41_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf60)
    del arg41_1
    del arg42_1
    buf61 = reinterpret_tensor(buf58, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf58  # reuse
    buf62 = reinterpret_tensor(buf59, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf59  # reuse
    buf63 = reinterpret_tensor(buf60, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf60  # reuse
    cpp_fused_9(c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf64 = aten._scaled_dot_product_flash_attention(buf61, buf62, buf63, scale=0.125)
    buf65 = buf64[0]
    del buf64
    buf72 = reinterpret_tensor(buf63, (512, 1024), (1024, 1), 0); del buf63  # reuse
    # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg44_1, reinterpret_tensor(buf65, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg43_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf72)
    del arg43_1
    del arg44_1
    buf73 = buf55; del buf55  # reuse
    buf74 = buf54; del buf54  # reuse
    buf76 = reinterpret_tensor(buf65, (1, 512, 1024), (524288, 1024, 1), 0); del buf65  # reuse
    cpp_fused_add_native_layer_norm_10(c_void_p(buf53.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf76.data_ptr()))
    del arg45_1
    del arg46_1
    buf77 = reinterpret_tensor(buf51, (512, 4096), (4096, 1), 0); del buf51  # reuse
    # Source Nodes: [hidden_states_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg48_1, reinterpret_tensor(buf76, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg47_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf77)
    del arg47_1
    del arg48_1
    buf78 = reinterpret_tensor(buf77, (1, 512, 4096), (2097152, 4096, 1), 0); del buf77  # reuse
    cpp_fused_gelu_11(c_void_p(buf78.data_ptr()))
    buf79 = reinterpret_tensor(buf76, (512, 1024), (1024, 1), 0); del buf76  # reuse
    # Source Nodes: [hidden_states_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg50_1, reinterpret_tensor(buf78, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg49_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf79)
    del arg49_1
    del arg50_1
    buf80 = buf74; del buf74  # reuse
    buf81 = buf73; del buf73  # reuse
    buf83 = reinterpret_tensor(buf62, (1, 512, 1024), (524288, 1024, 1), 0); del buf62  # reuse
    cpp_fused_add_native_layer_norm_12(c_void_p(buf53.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()))
    del arg51_1
    del arg52_1
    buf84 = reinterpret_tensor(buf61, (512, 1024), (1024, 1), 0); del buf61  # reuse
    # Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg54_1, reinterpret_tensor(buf83, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg53_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf84)
    del arg53_1
    del arg54_1
    buf85 = reinterpret_tensor(buf57, (512, 1024), (1024, 1), 0); del buf57  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_3_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg56_1, reinterpret_tensor(buf83, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg55_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf85)
    del arg55_1
    del arg56_1
    buf86 = reinterpret_tensor(buf0, (512, 1024), (1024, 1), 0); del buf0  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_3_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg58_1, reinterpret_tensor(buf83, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf86)
    del arg57_1
    del arg58_1
    del buf83
    buf87 = reinterpret_tensor(buf84, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf84  # reuse
    buf88 = reinterpret_tensor(buf85, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf85  # reuse
    buf89 = reinterpret_tensor(buf86, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf86  # reuse
    cpp_fused_13(c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf90 = aten._scaled_dot_product_flash_attention(buf87, buf88, buf89, scale=0.125)
    del buf87
    buf91 = buf90[0]
    del buf90
    buf98 = reinterpret_tensor(buf89, (512, 1024), (1024, 1), 0); del buf89  # reuse
    # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg60_1, reinterpret_tensor(buf91, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg59_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf98)
    del arg59_1
    del arg60_1
    buf99 = buf81; del buf81  # reuse
    buf100 = buf80; del buf80  # reuse
    buf102 = reinterpret_tensor(buf91, (1, 512, 1024), (524288, 1024, 1), 0); del buf91  # reuse
    cpp_fused_add_native_layer_norm_14(c_void_p(buf53.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()))
    del arg61_1
    del arg62_1
    buf103 = reinterpret_tensor(buf78, (512, 4096), (4096, 1), 0); del buf78  # reuse
    # Source Nodes: [hidden_states_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg64_1, reinterpret_tensor(buf102, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg63_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf103)
    del arg63_1
    del arg64_1
    buf104 = reinterpret_tensor(buf103, (1, 512, 4096), (2097152, 4096, 1), 0); del buf103  # reuse
    cpp_fused_gelu_15(c_void_p(buf104.data_ptr()))
    buf105 = reinterpret_tensor(buf102, (512, 1024), (1024, 1), 0); del buf102  # reuse
    # Source Nodes: [hidden_states_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg66_1, reinterpret_tensor(buf104, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg65_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf105)
    del arg65_1
    del arg66_1
    buf106 = reinterpret_tensor(buf105, (1, 512, 1024), (524288, 1024, 1), 0); del buf105  # reuse
    buf107 = buf99; del buf99  # reuse
    buf108 = buf100; del buf100  # reuse
    buf110 = reinterpret_tensor(buf88, (1, 512, 1024), (524288, 1024, 1), 0); del buf88  # reuse
    cpp_fused_add_native_layer_norm_16(c_void_p(buf106.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf110.data_ptr()))
    del arg67_1
    del arg68_1
    buf111 = buf98; del buf98  # reuse
    # Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg70_1, reinterpret_tensor(buf110, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg69_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf111)
    del arg69_1
    del arg70_1
    buf112 = buf79; del buf79  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_4_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg72_1, reinterpret_tensor(buf110, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf112)
    del arg71_1
    del arg72_1
    buf113 = buf72; del buf72  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_4_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg74_1, reinterpret_tensor(buf110, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf113)
    del arg73_1
    del arg74_1
    buf114 = reinterpret_tensor(buf111, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf111  # reuse
    buf115 = reinterpret_tensor(buf112, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf112  # reuse
    buf116 = reinterpret_tensor(buf113, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf113  # reuse
    cpp_fused_17(c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf117 = aten._scaled_dot_product_flash_attention(buf114, buf115, buf116, scale=0.125)
    buf118 = buf117[0]
    del buf117
    buf125 = reinterpret_tensor(buf116, (512, 1024), (1024, 1), 0); del buf116  # reuse
    # Source Nodes: [hidden_states_28], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg76_1, reinterpret_tensor(buf118, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg75_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf125)
    del arg75_1
    del arg76_1
    buf126 = buf108; del buf108  # reuse
    buf127 = buf107; del buf107  # reuse
    buf129 = reinterpret_tensor(buf118, (1, 512, 1024), (524288, 1024, 1), 0); del buf118  # reuse
    cpp_fused_add_native_layer_norm_18(c_void_p(buf106.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()))
    del arg77_1
    del arg78_1
    buf130 = reinterpret_tensor(buf104, (512, 4096), (4096, 1), 0); del buf104  # reuse
    # Source Nodes: [hidden_states_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg80_1, reinterpret_tensor(buf129, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg79_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf130)
    del arg79_1
    del arg80_1
    buf131 = reinterpret_tensor(buf130, (1, 512, 4096), (2097152, 4096, 1), 0); del buf130  # reuse
    cpp_fused_gelu_19(c_void_p(buf131.data_ptr()))
    buf132 = reinterpret_tensor(buf129, (512, 1024), (1024, 1), 0); del buf129  # reuse
    # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg82_1, reinterpret_tensor(buf131, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg81_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf132)
    del arg81_1
    del arg82_1
    buf133 = buf127; del buf127  # reuse
    buf134 = buf126; del buf126  # reuse
    buf136 = reinterpret_tensor(buf115, (1, 512, 1024), (524288, 1024, 1), 0); del buf115  # reuse
    cpp_fused_add_native_layer_norm_20(c_void_p(buf106.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()))
    del arg83_1
    del arg84_1
    buf137 = reinterpret_tensor(buf114, (512, 1024), (1024, 1), 0); del buf114  # reuse
    # Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg86_1, reinterpret_tensor(buf136, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg85_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf137)
    del arg85_1
    del arg86_1
    buf138 = reinterpret_tensor(buf110, (512, 1024), (1024, 1), 0); del buf110  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_5_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg88_1, reinterpret_tensor(buf136, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg87_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf138)
    del arg87_1
    del arg88_1
    buf139 = reinterpret_tensor(buf53, (512, 1024), (1024, 1), 0); del buf53  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_5_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg90_1, reinterpret_tensor(buf136, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg89_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf139)
    del arg89_1
    del arg90_1
    del buf136
    buf140 = reinterpret_tensor(buf137, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf137  # reuse
    buf141 = reinterpret_tensor(buf138, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf138  # reuse
    buf142 = reinterpret_tensor(buf139, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf139  # reuse
    cpp_fused_21(c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf143 = aten._scaled_dot_product_flash_attention(buf140, buf141, buf142, scale=0.125)
    del buf140
    buf144 = buf143[0]
    del buf143
    buf151 = reinterpret_tensor(buf142, (512, 1024), (1024, 1), 0); del buf142  # reuse
    # Source Nodes: [hidden_states_35], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg92_1, reinterpret_tensor(buf144, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg91_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf151)
    del arg91_1
    del arg92_1
    buf152 = buf134; del buf134  # reuse
    buf153 = buf133; del buf133  # reuse
    buf155 = reinterpret_tensor(buf144, (1, 512, 1024), (524288, 1024, 1), 0); del buf144  # reuse
    cpp_fused_add_native_layer_norm_22(c_void_p(buf106.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf155.data_ptr()))
    del arg93_1
    del arg94_1
    buf156 = reinterpret_tensor(buf131, (512, 4096), (4096, 1), 0); del buf131  # reuse
    # Source Nodes: [hidden_states_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg96_1, reinterpret_tensor(buf155, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg95_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf156)
    del arg95_1
    del arg96_1
    buf157 = reinterpret_tensor(buf156, (1, 512, 4096), (2097152, 4096, 1), 0); del buf156  # reuse
    cpp_fused_gelu_23(c_void_p(buf157.data_ptr()))
    buf158 = reinterpret_tensor(buf155, (512, 1024), (1024, 1), 0); del buf155  # reuse
    # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg98_1, reinterpret_tensor(buf157, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg97_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf158)
    del arg97_1
    del arg98_1
    buf159 = reinterpret_tensor(buf158, (1, 512, 1024), (524288, 1024, 1), 0); del buf158  # reuse
    buf160 = buf153; del buf153  # reuse
    buf161 = buf152; del buf152  # reuse
    buf163 = reinterpret_tensor(buf141, (1, 512, 1024), (524288, 1024, 1), 0); del buf141  # reuse
    cpp_fused_add_native_layer_norm_24(c_void_p(buf159.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()))
    del arg100_1
    del arg99_1
    buf164 = buf151; del buf151  # reuse
    # Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg102_1, reinterpret_tensor(buf163, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg101_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf164)
    del arg101_1
    del arg102_1
    buf165 = buf132; del buf132  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_6_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg104_1, reinterpret_tensor(buf163, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg103_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf165)
    del arg103_1
    del arg104_1
    buf166 = buf125; del buf125  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_6_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg106_1, reinterpret_tensor(buf163, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg105_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf166)
    del arg105_1
    del arg106_1
    buf167 = reinterpret_tensor(buf164, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf164  # reuse
    buf168 = reinterpret_tensor(buf165, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf165  # reuse
    buf169 = reinterpret_tensor(buf166, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf166  # reuse
    cpp_fused_25(c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf170 = aten._scaled_dot_product_flash_attention(buf167, buf168, buf169, scale=0.125)
    buf171 = buf170[0]
    del buf170
    buf178 = reinterpret_tensor(buf169, (512, 1024), (1024, 1), 0); del buf169  # reuse
    # Source Nodes: [hidden_states_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg108_1, reinterpret_tensor(buf171, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg107_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf178)
    del arg107_1
    del arg108_1
    buf179 = buf161; del buf161  # reuse
    buf180 = buf160; del buf160  # reuse
    buf182 = reinterpret_tensor(buf171, (1, 512, 1024), (524288, 1024, 1), 0); del buf171  # reuse
    cpp_fused_add_native_layer_norm_26(c_void_p(buf159.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()))
    del arg109_1
    del arg110_1
    buf183 = reinterpret_tensor(buf157, (512, 4096), (4096, 1), 0); del buf157  # reuse
    # Source Nodes: [hidden_states_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg112_1, reinterpret_tensor(buf182, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg111_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf183)
    del arg111_1
    del arg112_1
    buf184 = reinterpret_tensor(buf183, (1, 512, 4096), (2097152, 4096, 1), 0); del buf183  # reuse
    cpp_fused_gelu_27(c_void_p(buf184.data_ptr()))
    buf185 = reinterpret_tensor(buf182, (512, 1024), (1024, 1), 0); del buf182  # reuse
    # Source Nodes: [hidden_states_46], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg114_1, reinterpret_tensor(buf184, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg113_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf185)
    del arg113_1
    del arg114_1
    buf186 = buf180; del buf180  # reuse
    buf187 = buf179; del buf179  # reuse
    buf189 = reinterpret_tensor(buf168, (1, 512, 1024), (524288, 1024, 1), 0); del buf168  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf159.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf189.data_ptr()))
    del arg115_1
    del arg116_1
    buf190 = reinterpret_tensor(buf167, (512, 1024), (1024, 1), 0); del buf167  # reuse
    # Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg118_1, reinterpret_tensor(buf189, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg117_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf190)
    del arg117_1
    del arg118_1
    buf191 = reinterpret_tensor(buf163, (512, 1024), (1024, 1), 0); del buf163  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_7_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg120_1, reinterpret_tensor(buf189, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg119_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf191)
    del arg119_1
    del arg120_1
    buf192 = reinterpret_tensor(buf106, (512, 1024), (1024, 1), 0); del buf106  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_7_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg122_1, reinterpret_tensor(buf189, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg121_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf192)
    del arg121_1
    del arg122_1
    del buf189
    buf193 = reinterpret_tensor(buf190, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf190  # reuse
    buf194 = reinterpret_tensor(buf191, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf191  # reuse
    buf195 = reinterpret_tensor(buf192, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf192  # reuse
    cpp_fused_29(c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf196 = aten._scaled_dot_product_flash_attention(buf193, buf194, buf195, scale=0.125)
    del buf193
    buf197 = buf196[0]
    del buf196
    buf204 = reinterpret_tensor(buf195, (512, 1024), (1024, 1), 0); del buf195  # reuse
    # Source Nodes: [hidden_states_49], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg124_1, reinterpret_tensor(buf197, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg123_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf204)
    del arg123_1
    del arg124_1
    buf205 = buf187; del buf187  # reuse
    buf206 = buf186; del buf186  # reuse
    buf208 = reinterpret_tensor(buf197, (1, 512, 1024), (524288, 1024, 1), 0); del buf197  # reuse
    cpp_fused_add_native_layer_norm_30(c_void_p(buf159.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf208.data_ptr()))
    del arg125_1
    del arg126_1
    buf209 = reinterpret_tensor(buf184, (512, 4096), (4096, 1), 0); del buf184  # reuse
    # Source Nodes: [hidden_states_51], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg128_1, reinterpret_tensor(buf208, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg127_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf209)
    del arg127_1
    del arg128_1
    buf210 = reinterpret_tensor(buf209, (1, 512, 4096), (2097152, 4096, 1), 0); del buf209  # reuse
    cpp_fused_gelu_31(c_void_p(buf210.data_ptr()))
    buf211 = reinterpret_tensor(buf208, (512, 1024), (1024, 1), 0); del buf208  # reuse
    # Source Nodes: [hidden_states_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg130_1, reinterpret_tensor(buf210, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg129_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf211)
    del arg129_1
    del arg130_1
    buf212 = reinterpret_tensor(buf211, (1, 512, 1024), (524288, 1024, 1), 0); del buf211  # reuse
    buf213 = buf206; del buf206  # reuse
    buf214 = buf205; del buf205  # reuse
    buf216 = reinterpret_tensor(buf194, (1, 512, 1024), (524288, 1024, 1), 0); del buf194  # reuse
    cpp_fused_add_native_layer_norm_32(c_void_p(buf212.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()))
    del arg131_1
    del arg132_1
    buf217 = buf204; del buf204  # reuse
    # Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg134_1, reinterpret_tensor(buf216, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg133_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf217)
    del arg133_1
    del arg134_1
    buf218 = buf185; del buf185  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_8_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg136_1, reinterpret_tensor(buf216, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg135_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf218)
    del arg135_1
    del arg136_1
    buf219 = buf178; del buf178  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_8_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf216, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf219)
    del arg137_1
    del arg138_1
    buf220 = reinterpret_tensor(buf217, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf217  # reuse
    buf221 = reinterpret_tensor(buf218, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf218  # reuse
    buf222 = reinterpret_tensor(buf219, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf219  # reuse
    cpp_fused_33(c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf223 = aten._scaled_dot_product_flash_attention(buf220, buf221, buf222, scale=0.125)
    buf224 = buf223[0]
    del buf223
    buf231 = reinterpret_tensor(buf222, (512, 1024), (1024, 1), 0); del buf222  # reuse
    # Source Nodes: [hidden_states_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg140_1, reinterpret_tensor(buf224, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg139_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf231)
    del arg139_1
    del arg140_1
    buf232 = buf214; del buf214  # reuse
    buf233 = buf213; del buf213  # reuse
    buf235 = reinterpret_tensor(buf224, (1, 512, 1024), (524288, 1024, 1), 0); del buf224  # reuse
    cpp_fused_add_native_layer_norm_34(c_void_p(buf212.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf235.data_ptr()))
    del arg141_1
    del arg142_1
    buf236 = reinterpret_tensor(buf210, (512, 4096), (4096, 1), 0); del buf210  # reuse
    # Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg144_1, reinterpret_tensor(buf235, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg143_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf236)
    del arg143_1
    del arg144_1
    buf237 = reinterpret_tensor(buf236, (1, 512, 4096), (2097152, 4096, 1), 0); del buf236  # reuse
    cpp_fused_gelu_35(c_void_p(buf237.data_ptr()))
    buf238 = reinterpret_tensor(buf235, (512, 1024), (1024, 1), 0); del buf235  # reuse
    # Source Nodes: [hidden_states_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg146_1, reinterpret_tensor(buf237, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg145_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf238)
    del arg145_1
    del arg146_1
    buf239 = buf233; del buf233  # reuse
    buf240 = buf232; del buf232  # reuse
    buf242 = reinterpret_tensor(buf221, (1, 512, 1024), (524288, 1024, 1), 0); del buf221  # reuse
    cpp_fused_add_native_layer_norm_36(c_void_p(buf212.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf242.data_ptr()))
    del arg147_1
    del arg148_1
    buf243 = reinterpret_tensor(buf220, (512, 1024), (1024, 1), 0); del buf220  # reuse
    # Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg150_1, reinterpret_tensor(buf242, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg149_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf243)
    del arg149_1
    del arg150_1
    buf244 = reinterpret_tensor(buf216, (512, 1024), (1024, 1), 0); del buf216  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_9_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg152_1, reinterpret_tensor(buf242, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg151_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf244)
    del arg151_1
    del arg152_1
    buf245 = reinterpret_tensor(buf159, (512, 1024), (1024, 1), 0); del buf159  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_9_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg154_1, reinterpret_tensor(buf242, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg153_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf245)
    del arg153_1
    del arg154_1
    del buf242
    buf246 = reinterpret_tensor(buf243, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf243  # reuse
    buf247 = reinterpret_tensor(buf244, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf244  # reuse
    buf248 = reinterpret_tensor(buf245, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf245  # reuse
    cpp_fused_37(c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf249 = aten._scaled_dot_product_flash_attention(buf246, buf247, buf248, scale=0.125)
    del buf246
    buf250 = buf249[0]
    del buf249
    buf257 = reinterpret_tensor(buf248, (512, 1024), (1024, 1), 0); del buf248  # reuse
    # Source Nodes: [hidden_states_63], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg156_1, reinterpret_tensor(buf250, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg155_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf257)
    del arg155_1
    del arg156_1
    buf258 = buf240; del buf240  # reuse
    buf259 = buf239; del buf239  # reuse
    buf261 = reinterpret_tensor(buf250, (1, 512, 1024), (524288, 1024, 1), 0); del buf250  # reuse
    cpp_fused_add_native_layer_norm_38(c_void_p(buf212.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()))
    del arg157_1
    del arg158_1
    buf262 = reinterpret_tensor(buf237, (512, 4096), (4096, 1), 0); del buf237  # reuse
    # Source Nodes: [hidden_states_65], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg160_1, reinterpret_tensor(buf261, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg159_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf262)
    del arg159_1
    del arg160_1
    buf263 = reinterpret_tensor(buf262, (1, 512, 4096), (2097152, 4096, 1), 0); del buf262  # reuse
    cpp_fused_gelu_39(c_void_p(buf263.data_ptr()))
    buf264 = reinterpret_tensor(buf261, (512, 1024), (1024, 1), 0); del buf261  # reuse
    # Source Nodes: [hidden_states_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg162_1, reinterpret_tensor(buf263, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg161_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf264)
    del arg161_1
    del arg162_1
    buf265 = reinterpret_tensor(buf264, (1, 512, 1024), (524288, 1024, 1), 0); del buf264  # reuse
    buf266 = buf259; del buf259  # reuse
    buf267 = buf258; del buf258  # reuse
    buf269 = reinterpret_tensor(buf247, (1, 512, 1024), (524288, 1024, 1), 0); del buf247  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf265.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf269.data_ptr()))
    del arg163_1
    del arg164_1
    buf270 = buf257; del buf257  # reuse
    # Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg166_1, reinterpret_tensor(buf269, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg165_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf270)
    del arg165_1
    del arg166_1
    buf271 = buf238; del buf238  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_10_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg168_1, reinterpret_tensor(buf269, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg167_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf271)
    del arg167_1
    del arg168_1
    buf272 = buf231; del buf231  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_10_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg170_1, reinterpret_tensor(buf269, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg169_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf272)
    del arg169_1
    del arg170_1
    buf273 = reinterpret_tensor(buf270, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf270  # reuse
    buf274 = reinterpret_tensor(buf271, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf271  # reuse
    buf275 = reinterpret_tensor(buf272, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf272  # reuse
    cpp_fused_41(c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf276 = aten._scaled_dot_product_flash_attention(buf273, buf274, buf275, scale=0.125)
    buf277 = buf276[0]
    del buf276
    buf284 = reinterpret_tensor(buf275, (512, 1024), (1024, 1), 0); del buf275  # reuse
    # Source Nodes: [hidden_states_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg172_1, reinterpret_tensor(buf277, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf284)
    del arg171_1
    del arg172_1
    buf285 = buf267; del buf267  # reuse
    buf286 = buf266; del buf266  # reuse
    buf288 = reinterpret_tensor(buf277, (1, 512, 1024), (524288, 1024, 1), 0); del buf277  # reuse
    cpp_fused_add_native_layer_norm_42(c_void_p(buf265.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf288.data_ptr()))
    del arg173_1
    del arg174_1
    buf289 = reinterpret_tensor(buf263, (512, 4096), (4096, 1), 0); del buf263  # reuse
    # Source Nodes: [hidden_states_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg176_1, reinterpret_tensor(buf288, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg175_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf289)
    del arg175_1
    del arg176_1
    buf290 = reinterpret_tensor(buf289, (1, 512, 4096), (2097152, 4096, 1), 0); del buf289  # reuse
    cpp_fused_gelu_43(c_void_p(buf290.data_ptr()))
    buf291 = reinterpret_tensor(buf288, (512, 1024), (1024, 1), 0); del buf288  # reuse
    # Source Nodes: [hidden_states_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg178_1, reinterpret_tensor(buf290, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg177_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf291)
    del arg177_1
    del arg178_1
    buf292 = buf286; del buf286  # reuse
    buf293 = buf285; del buf285  # reuse
    buf295 = reinterpret_tensor(buf274, (1, 512, 1024), (524288, 1024, 1), 0); del buf274  # reuse
    cpp_fused_add_native_layer_norm_44(c_void_p(buf265.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf295.data_ptr()))
    del arg179_1
    del arg180_1
    buf296 = reinterpret_tensor(buf273, (512, 1024), (1024, 1), 0); del buf273  # reuse
    # Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg182_1, reinterpret_tensor(buf295, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg181_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf296)
    del arg181_1
    del arg182_1
    buf297 = reinterpret_tensor(buf269, (512, 1024), (1024, 1), 0); del buf269  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_11_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg184_1, reinterpret_tensor(buf295, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg183_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf297)
    del arg183_1
    del arg184_1
    buf298 = reinterpret_tensor(buf212, (512, 1024), (1024, 1), 0); del buf212  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_11_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg186_1, reinterpret_tensor(buf295, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg185_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf298)
    del arg185_1
    del arg186_1
    del buf295
    buf299 = reinterpret_tensor(buf296, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf296  # reuse
    buf300 = reinterpret_tensor(buf297, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf297  # reuse
    buf301 = reinterpret_tensor(buf298, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf298  # reuse
    cpp_fused_45(c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf302 = aten._scaled_dot_product_flash_attention(buf299, buf300, buf301, scale=0.125)
    del buf299
    buf303 = buf302[0]
    del buf302
    buf310 = reinterpret_tensor(buf301, (512, 1024), (1024, 1), 0); del buf301  # reuse
    # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg188_1, reinterpret_tensor(buf303, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg187_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf310)
    del arg187_1
    del arg188_1
    buf311 = buf293; del buf293  # reuse
    buf312 = buf292; del buf292  # reuse
    buf314 = reinterpret_tensor(buf303, (1, 512, 1024), (524288, 1024, 1), 0); del buf303  # reuse
    cpp_fused_add_native_layer_norm_46(c_void_p(buf265.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf314.data_ptr()))
    del arg189_1
    del arg190_1
    buf315 = reinterpret_tensor(buf290, (512, 4096), (4096, 1), 0); del buf290  # reuse
    # Source Nodes: [hidden_states_79], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg192_1, reinterpret_tensor(buf314, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg191_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf315)
    del arg191_1
    del arg192_1
    buf316 = reinterpret_tensor(buf315, (1, 512, 4096), (2097152, 4096, 1), 0); del buf315  # reuse
    cpp_fused_gelu_47(c_void_p(buf316.data_ptr()))
    buf317 = reinterpret_tensor(buf314, (512, 1024), (1024, 1), 0); del buf314  # reuse
    # Source Nodes: [hidden_states_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg194_1, reinterpret_tensor(buf316, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg193_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf317)
    del arg193_1
    del arg194_1
    buf318 = reinterpret_tensor(buf317, (1, 512, 1024), (524288, 1024, 1), 0); del buf317  # reuse
    buf319 = buf312; del buf312  # reuse
    buf320 = buf311; del buf311  # reuse
    buf322 = reinterpret_tensor(buf300, (1, 512, 1024), (524288, 1024, 1), 0); del buf300  # reuse
    cpp_fused_add_native_layer_norm_48(c_void_p(buf318.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf322.data_ptr()))
    del arg195_1
    del arg196_1
    buf323 = buf310; del buf310  # reuse
    # Source Nodes: [mixed_query_layer_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg198_1, reinterpret_tensor(buf322, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg197_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf323)
    del arg197_1
    del arg198_1
    buf324 = buf291; del buf291  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_12_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg200_1, reinterpret_tensor(buf322, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg199_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf324)
    del arg199_1
    del arg200_1
    buf325 = buf284; del buf284  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_12_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg202_1, reinterpret_tensor(buf322, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg201_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf325)
    del arg201_1
    del arg202_1
    buf326 = reinterpret_tensor(buf323, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf323  # reuse
    buf327 = reinterpret_tensor(buf324, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf324  # reuse
    buf328 = reinterpret_tensor(buf325, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf325  # reuse
    cpp_fused_49(c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf329 = aten._scaled_dot_product_flash_attention(buf326, buf327, buf328, scale=0.125)
    buf330 = buf329[0]
    del buf329
    buf337 = reinterpret_tensor(buf328, (512, 1024), (1024, 1), 0); del buf328  # reuse
    # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg204_1, reinterpret_tensor(buf330, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg203_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf337)
    del arg203_1
    del arg204_1
    buf338 = buf320; del buf320  # reuse
    buf339 = buf319; del buf319  # reuse
    buf341 = reinterpret_tensor(buf330, (1, 512, 1024), (524288, 1024, 1), 0); del buf330  # reuse
    cpp_fused_add_native_layer_norm_50(c_void_p(buf318.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf341.data_ptr()))
    del arg205_1
    del arg206_1
    buf342 = reinterpret_tensor(buf316, (512, 4096), (4096, 1), 0); del buf316  # reuse
    # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg208_1, reinterpret_tensor(buf341, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg207_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf342)
    del arg207_1
    del arg208_1
    buf343 = reinterpret_tensor(buf342, (1, 512, 4096), (2097152, 4096, 1), 0); del buf342  # reuse
    cpp_fused_gelu_51(c_void_p(buf343.data_ptr()))
    buf344 = reinterpret_tensor(buf341, (512, 1024), (1024, 1), 0); del buf341  # reuse
    # Source Nodes: [hidden_states_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg210_1, reinterpret_tensor(buf343, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg209_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf344)
    del arg209_1
    del arg210_1
    buf345 = buf339; del buf339  # reuse
    buf346 = buf338; del buf338  # reuse
    buf348 = reinterpret_tensor(buf327, (1, 512, 1024), (524288, 1024, 1), 0); del buf327  # reuse
    cpp_fused_add_native_layer_norm_52(c_void_p(buf318.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf348.data_ptr()))
    del arg211_1
    del arg212_1
    buf349 = reinterpret_tensor(buf326, (512, 1024), (1024, 1), 0); del buf326  # reuse
    # Source Nodes: [mixed_query_layer_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg214_1, reinterpret_tensor(buf348, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg213_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf349)
    del arg213_1
    del arg214_1
    buf350 = reinterpret_tensor(buf322, (512, 1024), (1024, 1), 0); del buf322  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_13_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg216_1, reinterpret_tensor(buf348, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg215_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf350)
    del arg215_1
    del arg216_1
    buf351 = reinterpret_tensor(buf265, (512, 1024), (1024, 1), 0); del buf265  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_13_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg218_1, reinterpret_tensor(buf348, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg217_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf351)
    del arg217_1
    del arg218_1
    del buf348
    buf352 = reinterpret_tensor(buf349, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf349  # reuse
    buf353 = reinterpret_tensor(buf350, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf350  # reuse
    buf354 = reinterpret_tensor(buf351, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf351  # reuse
    cpp_fused_53(c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf355 = aten._scaled_dot_product_flash_attention(buf352, buf353, buf354, scale=0.125)
    del buf352
    buf356 = buf355[0]
    del buf355
    buf363 = reinterpret_tensor(buf354, (512, 1024), (1024, 1), 0); del buf354  # reuse
    # Source Nodes: [hidden_states_91], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg220_1, reinterpret_tensor(buf356, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg219_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf363)
    del arg219_1
    del arg220_1
    buf364 = buf346; del buf346  # reuse
    buf365 = buf345; del buf345  # reuse
    buf367 = reinterpret_tensor(buf356, (1, 512, 1024), (524288, 1024, 1), 0); del buf356  # reuse
    cpp_fused_add_native_layer_norm_54(c_void_p(buf318.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf367.data_ptr()))
    del arg221_1
    del arg222_1
    buf368 = reinterpret_tensor(buf343, (512, 4096), (4096, 1), 0); del buf343  # reuse
    # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg224_1, reinterpret_tensor(buf367, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg223_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf368)
    del arg223_1
    del arg224_1
    buf369 = reinterpret_tensor(buf368, (1, 512, 4096), (2097152, 4096, 1), 0); del buf368  # reuse
    cpp_fused_gelu_55(c_void_p(buf369.data_ptr()))
    buf370 = reinterpret_tensor(buf367, (512, 1024), (1024, 1), 0); del buf367  # reuse
    # Source Nodes: [hidden_states_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg226_1, reinterpret_tensor(buf369, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg225_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf370)
    del arg225_1
    del arg226_1
    buf371 = reinterpret_tensor(buf370, (1, 512, 1024), (524288, 1024, 1), 0); del buf370  # reuse
    buf372 = buf365; del buf365  # reuse
    buf373 = buf364; del buf364  # reuse
    buf375 = reinterpret_tensor(buf353, (1, 512, 1024), (524288, 1024, 1), 0); del buf353  # reuse
    cpp_fused_add_native_layer_norm_56(c_void_p(buf371.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf375.data_ptr()))
    del arg227_1
    del arg228_1
    buf376 = buf363; del buf363  # reuse
    # Source Nodes: [mixed_query_layer_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg230_1, reinterpret_tensor(buf375, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg229_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf376)
    del arg229_1
    del arg230_1
    buf377 = buf344; del buf344  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_14_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg232_1, reinterpret_tensor(buf375, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg231_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf377)
    del arg231_1
    del arg232_1
    buf378 = buf337; del buf337  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_14_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg234_1, reinterpret_tensor(buf375, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg233_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf378)
    del arg233_1
    del arg234_1
    buf379 = reinterpret_tensor(buf376, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf376  # reuse
    buf380 = reinterpret_tensor(buf377, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf377  # reuse
    buf381 = reinterpret_tensor(buf378, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf378  # reuse
    cpp_fused_57(c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf382 = aten._scaled_dot_product_flash_attention(buf379, buf380, buf381, scale=0.125)
    buf383 = buf382[0]
    del buf382
    buf390 = reinterpret_tensor(buf381, (512, 1024), (1024, 1), 0); del buf381  # reuse
    # Source Nodes: [hidden_states_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg236_1, reinterpret_tensor(buf383, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg235_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf390)
    del arg235_1
    del arg236_1
    buf391 = buf373; del buf373  # reuse
    buf392 = buf372; del buf372  # reuse
    buf394 = reinterpret_tensor(buf383, (1, 512, 1024), (524288, 1024, 1), 0); del buf383  # reuse
    cpp_fused_add_native_layer_norm_58(c_void_p(buf371.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf394.data_ptr()))
    del arg237_1
    del arg238_1
    buf395 = reinterpret_tensor(buf369, (512, 4096), (4096, 1), 0); del buf369  # reuse
    # Source Nodes: [hidden_states_100], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg240_1, reinterpret_tensor(buf394, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg239_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf395)
    del arg239_1
    del arg240_1
    buf396 = reinterpret_tensor(buf395, (1, 512, 4096), (2097152, 4096, 1), 0); del buf395  # reuse
    cpp_fused_gelu_59(c_void_p(buf396.data_ptr()))
    buf397 = reinterpret_tensor(buf394, (512, 1024), (1024, 1), 0); del buf394  # reuse
    # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg242_1, reinterpret_tensor(buf396, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg241_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf397)
    del arg241_1
    del arg242_1
    buf398 = buf392; del buf392  # reuse
    buf399 = buf391; del buf391  # reuse
    buf401 = reinterpret_tensor(buf380, (1, 512, 1024), (524288, 1024, 1), 0); del buf380  # reuse
    cpp_fused_add_native_layer_norm_60(c_void_p(buf371.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf401.data_ptr()))
    del arg243_1
    del arg244_1
    buf402 = reinterpret_tensor(buf379, (512, 1024), (1024, 1), 0); del buf379  # reuse
    # Source Nodes: [mixed_query_layer_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg246_1, reinterpret_tensor(buf401, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg245_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf402)
    del arg245_1
    del arg246_1
    buf403 = reinterpret_tensor(buf375, (512, 1024), (1024, 1), 0); del buf375  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_15_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg248_1, reinterpret_tensor(buf401, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg247_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf403)
    del arg247_1
    del arg248_1
    buf404 = reinterpret_tensor(buf318, (512, 1024), (1024, 1), 0); del buf318  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_15_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg250_1, reinterpret_tensor(buf401, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg249_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf404)
    del arg249_1
    del arg250_1
    del buf401
    buf405 = reinterpret_tensor(buf402, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf402  # reuse
    buf406 = reinterpret_tensor(buf403, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf403  # reuse
    buf407 = reinterpret_tensor(buf404, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf404  # reuse
    cpp_fused_61(c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf408 = aten._scaled_dot_product_flash_attention(buf405, buf406, buf407, scale=0.125)
    del buf405
    buf409 = buf408[0]
    del buf408
    buf416 = reinterpret_tensor(buf407, (512, 1024), (1024, 1), 0); del buf407  # reuse
    # Source Nodes: [hidden_states_105], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg252_1, reinterpret_tensor(buf409, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg251_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf416)
    del arg251_1
    del arg252_1
    buf417 = buf399; del buf399  # reuse
    buf418 = buf398; del buf398  # reuse
    buf420 = reinterpret_tensor(buf409, (1, 512, 1024), (524288, 1024, 1), 0); del buf409  # reuse
    cpp_fused_add_native_layer_norm_62(c_void_p(buf371.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf420.data_ptr()))
    del arg253_1
    del arg254_1
    buf421 = reinterpret_tensor(buf396, (512, 4096), (4096, 1), 0); del buf396  # reuse
    # Source Nodes: [hidden_states_107], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg256_1, reinterpret_tensor(buf420, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg255_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf421)
    del arg255_1
    del arg256_1
    buf422 = reinterpret_tensor(buf421, (1, 512, 4096), (2097152, 4096, 1), 0); del buf421  # reuse
    cpp_fused_gelu_63(c_void_p(buf422.data_ptr()))
    buf423 = reinterpret_tensor(buf420, (512, 1024), (1024, 1), 0); del buf420  # reuse
    # Source Nodes: [hidden_states_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg258_1, reinterpret_tensor(buf422, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg257_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf423)
    del arg257_1
    del arg258_1
    buf424 = reinterpret_tensor(buf423, (1, 512, 1024), (524288, 1024, 1), 0); del buf423  # reuse
    buf425 = buf418; del buf418  # reuse
    buf426 = buf417; del buf417  # reuse
    buf428 = reinterpret_tensor(buf406, (1, 512, 1024), (524288, 1024, 1), 0); del buf406  # reuse
    cpp_fused_add_native_layer_norm_64(c_void_p(buf424.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf428.data_ptr()))
    del arg259_1
    del arg260_1
    buf429 = buf416; del buf416  # reuse
    # Source Nodes: [mixed_query_layer_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg262_1, reinterpret_tensor(buf428, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg261_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf429)
    del arg261_1
    del arg262_1
    buf430 = buf397; del buf397  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_16_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg264_1, reinterpret_tensor(buf428, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg263_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf430)
    del arg263_1
    del arg264_1
    buf431 = buf390; del buf390  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_16_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg266_1, reinterpret_tensor(buf428, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg265_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf431)
    del arg265_1
    del arg266_1
    buf432 = reinterpret_tensor(buf429, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf429  # reuse
    buf433 = reinterpret_tensor(buf430, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf430  # reuse
    buf434 = reinterpret_tensor(buf431, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf431  # reuse
    cpp_fused_65(c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf435 = aten._scaled_dot_product_flash_attention(buf432, buf433, buf434, scale=0.125)
    buf436 = buf435[0]
    del buf435
    buf443 = reinterpret_tensor(buf434, (512, 1024), (1024, 1), 0); del buf434  # reuse
    # Source Nodes: [hidden_states_112], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg268_1, reinterpret_tensor(buf436, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg267_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf443)
    del arg267_1
    del arg268_1
    buf444 = buf426; del buf426  # reuse
    buf445 = buf425; del buf425  # reuse
    buf447 = reinterpret_tensor(buf436, (1, 512, 1024), (524288, 1024, 1), 0); del buf436  # reuse
    cpp_fused_add_native_layer_norm_66(c_void_p(buf424.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf447.data_ptr()))
    del arg269_1
    del arg270_1
    buf448 = reinterpret_tensor(buf422, (512, 4096), (4096, 1), 0); del buf422  # reuse
    # Source Nodes: [hidden_states_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg272_1, reinterpret_tensor(buf447, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg271_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf448)
    del arg271_1
    del arg272_1
    buf449 = reinterpret_tensor(buf448, (1, 512, 4096), (2097152, 4096, 1), 0); del buf448  # reuse
    cpp_fused_gelu_67(c_void_p(buf449.data_ptr()))
    buf450 = reinterpret_tensor(buf447, (512, 1024), (1024, 1), 0); del buf447  # reuse
    # Source Nodes: [hidden_states_116], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg274_1, reinterpret_tensor(buf449, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg273_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf450)
    del arg273_1
    del arg274_1
    buf451 = buf445; del buf445  # reuse
    buf452 = buf444; del buf444  # reuse
    buf454 = reinterpret_tensor(buf433, (1, 512, 1024), (524288, 1024, 1), 0); del buf433  # reuse
    cpp_fused_add_native_layer_norm_68(c_void_p(buf424.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf454.data_ptr()))
    del arg275_1
    del arg276_1
    buf455 = reinterpret_tensor(buf432, (512, 1024), (1024, 1), 0); del buf432  # reuse
    # Source Nodes: [mixed_query_layer_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg278_1, reinterpret_tensor(buf454, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg277_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf455)
    del arg277_1
    del arg278_1
    buf456 = reinterpret_tensor(buf428, (512, 1024), (1024, 1), 0); del buf428  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_17_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg280_1, reinterpret_tensor(buf454, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg279_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf456)
    del arg279_1
    del arg280_1
    buf457 = reinterpret_tensor(buf371, (512, 1024), (1024, 1), 0); del buf371  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_17_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg282_1, reinterpret_tensor(buf454, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg281_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf457)
    del arg281_1
    del arg282_1
    del buf454
    buf458 = reinterpret_tensor(buf455, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf455  # reuse
    buf459 = reinterpret_tensor(buf456, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf456  # reuse
    buf460 = reinterpret_tensor(buf457, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf457  # reuse
    cpp_fused_69(c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf461 = aten._scaled_dot_product_flash_attention(buf458, buf459, buf460, scale=0.125)
    del buf458
    buf462 = buf461[0]
    del buf461
    buf469 = reinterpret_tensor(buf460, (512, 1024), (1024, 1), 0); del buf460  # reuse
    # Source Nodes: [hidden_states_119], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg284_1, reinterpret_tensor(buf462, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg283_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf469)
    del arg283_1
    del arg284_1
    buf470 = buf452; del buf452  # reuse
    buf471 = buf451; del buf451  # reuse
    buf473 = reinterpret_tensor(buf462, (1, 512, 1024), (524288, 1024, 1), 0); del buf462  # reuse
    cpp_fused_add_native_layer_norm_70(c_void_p(buf424.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf473.data_ptr()))
    del arg285_1
    del arg286_1
    buf474 = reinterpret_tensor(buf449, (512, 4096), (4096, 1), 0); del buf449  # reuse
    # Source Nodes: [hidden_states_121], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg288_1, reinterpret_tensor(buf473, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg287_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf474)
    del arg287_1
    del arg288_1
    buf475 = reinterpret_tensor(buf474, (1, 512, 4096), (2097152, 4096, 1), 0); del buf474  # reuse
    cpp_fused_gelu_71(c_void_p(buf475.data_ptr()))
    buf476 = reinterpret_tensor(buf473, (512, 1024), (1024, 1), 0); del buf473  # reuse
    # Source Nodes: [hidden_states_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg290_1, reinterpret_tensor(buf475, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg289_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf476)
    del arg289_1
    del arg290_1
    buf477 = reinterpret_tensor(buf476, (1, 512, 1024), (524288, 1024, 1), 0); del buf476  # reuse
    buf478 = buf471; del buf471  # reuse
    buf479 = buf470; del buf470  # reuse
    buf481 = reinterpret_tensor(buf459, (1, 512, 1024), (524288, 1024, 1), 0); del buf459  # reuse
    cpp_fused_add_native_layer_norm_72(c_void_p(buf477.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf481.data_ptr()))
    del arg291_1
    del arg292_1
    buf482 = buf469; del buf469  # reuse
    # Source Nodes: [mixed_query_layer_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg294_1, reinterpret_tensor(buf481, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg293_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf482)
    del arg293_1
    del arg294_1
    buf483 = buf450; del buf450  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_18_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg296_1, reinterpret_tensor(buf481, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg295_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf483)
    del arg295_1
    del arg296_1
    buf484 = buf443; del buf443  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_18_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg298_1, reinterpret_tensor(buf481, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg297_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf484)
    del arg297_1
    del arg298_1
    buf485 = reinterpret_tensor(buf482, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf482  # reuse
    buf486 = reinterpret_tensor(buf483, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf483  # reuse
    buf487 = reinterpret_tensor(buf484, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf484  # reuse
    cpp_fused_73(c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf488 = aten._scaled_dot_product_flash_attention(buf485, buf486, buf487, scale=0.125)
    buf489 = buf488[0]
    del buf488
    buf496 = reinterpret_tensor(buf487, (512, 1024), (1024, 1), 0); del buf487  # reuse
    # Source Nodes: [hidden_states_126], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg300_1, reinterpret_tensor(buf489, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg299_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf496)
    del arg299_1
    del arg300_1
    buf497 = buf479; del buf479  # reuse
    buf498 = buf478; del buf478  # reuse
    buf500 = reinterpret_tensor(buf489, (1, 512, 1024), (524288, 1024, 1), 0); del buf489  # reuse
    cpp_fused_add_native_layer_norm_74(c_void_p(buf477.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf500.data_ptr()))
    del arg301_1
    del arg302_1
    buf501 = reinterpret_tensor(buf475, (512, 4096), (4096, 1), 0); del buf475  # reuse
    # Source Nodes: [hidden_states_128], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg304_1, reinterpret_tensor(buf500, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg303_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf501)
    del arg303_1
    del arg304_1
    buf502 = reinterpret_tensor(buf501, (1, 512, 4096), (2097152, 4096, 1), 0); del buf501  # reuse
    cpp_fused_gelu_75(c_void_p(buf502.data_ptr()))
    buf503 = reinterpret_tensor(buf500, (512, 1024), (1024, 1), 0); del buf500  # reuse
    # Source Nodes: [hidden_states_130], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg306_1, reinterpret_tensor(buf502, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg305_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf503)
    del arg305_1
    del arg306_1
    buf504 = buf498; del buf498  # reuse
    buf505 = buf497; del buf497  # reuse
    buf507 = reinterpret_tensor(buf486, (1, 512, 1024), (524288, 1024, 1), 0); del buf486  # reuse
    cpp_fused_add_native_layer_norm_76(c_void_p(buf477.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf507.data_ptr()))
    del arg307_1
    del arg308_1
    buf508 = reinterpret_tensor(buf485, (512, 1024), (1024, 1), 0); del buf485  # reuse
    # Source Nodes: [mixed_query_layer_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg310_1, reinterpret_tensor(buf507, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg309_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf508)
    del arg309_1
    del arg310_1
    buf509 = reinterpret_tensor(buf481, (512, 1024), (1024, 1), 0); del buf481  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_19_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg312_1, reinterpret_tensor(buf507, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg311_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf509)
    del arg311_1
    del arg312_1
    buf510 = reinterpret_tensor(buf424, (512, 1024), (1024, 1), 0); del buf424  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_19_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg314_1, reinterpret_tensor(buf507, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg313_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf510)
    del arg313_1
    del arg314_1
    del buf507
    buf511 = reinterpret_tensor(buf508, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf508  # reuse
    buf512 = reinterpret_tensor(buf509, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf509  # reuse
    buf513 = reinterpret_tensor(buf510, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf510  # reuse
    cpp_fused_77(c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf514 = aten._scaled_dot_product_flash_attention(buf511, buf512, buf513, scale=0.125)
    del buf511
    buf515 = buf514[0]
    del buf514
    buf522 = reinterpret_tensor(buf513, (512, 1024), (1024, 1), 0); del buf513  # reuse
    # Source Nodes: [hidden_states_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg316_1, reinterpret_tensor(buf515, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg315_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf522)
    del arg315_1
    del arg316_1
    buf523 = buf505; del buf505  # reuse
    buf524 = buf504; del buf504  # reuse
    buf526 = reinterpret_tensor(buf515, (1, 512, 1024), (524288, 1024, 1), 0); del buf515  # reuse
    cpp_fused_add_native_layer_norm_78(c_void_p(buf477.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf526.data_ptr()))
    del arg317_1
    del arg318_1
    buf527 = reinterpret_tensor(buf502, (512, 4096), (4096, 1), 0); del buf502  # reuse
    # Source Nodes: [hidden_states_135], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg320_1, reinterpret_tensor(buf526, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg319_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf527)
    del arg319_1
    del arg320_1
    buf528 = reinterpret_tensor(buf527, (1, 512, 4096), (2097152, 4096, 1), 0); del buf527  # reuse
    cpp_fused_gelu_79(c_void_p(buf528.data_ptr()))
    buf529 = reinterpret_tensor(buf526, (512, 1024), (1024, 1), 0); del buf526  # reuse
    # Source Nodes: [hidden_states_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg322_1, reinterpret_tensor(buf528, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg321_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf529)
    del arg321_1
    del arg322_1
    buf530 = reinterpret_tensor(buf529, (1, 512, 1024), (524288, 1024, 1), 0); del buf529  # reuse
    buf531 = buf524; del buf524  # reuse
    buf532 = buf523; del buf523  # reuse
    buf534 = reinterpret_tensor(buf512, (1, 512, 1024), (524288, 1024, 1), 0); del buf512  # reuse
    cpp_fused_add_native_layer_norm_80(c_void_p(buf530.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf534.data_ptr()))
    del arg323_1
    del arg324_1
    buf535 = buf522; del buf522  # reuse
    # Source Nodes: [mixed_query_layer_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg326_1, reinterpret_tensor(buf534, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg325_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf535)
    del arg325_1
    del arg326_1
    buf536 = buf503; del buf503  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_20_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg328_1, reinterpret_tensor(buf534, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg327_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf536)
    del arg327_1
    del arg328_1
    buf537 = buf496; del buf496  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_20_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg330_1, reinterpret_tensor(buf534, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg329_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf537)
    del arg329_1
    del arg330_1
    buf538 = reinterpret_tensor(buf535, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf535  # reuse
    buf539 = reinterpret_tensor(buf536, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf536  # reuse
    buf540 = reinterpret_tensor(buf537, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf537  # reuse
    cpp_fused_81(c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf540.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf541 = aten._scaled_dot_product_flash_attention(buf538, buf539, buf540, scale=0.125)
    buf542 = buf541[0]
    del buf541
    buf549 = reinterpret_tensor(buf540, (512, 1024), (1024, 1), 0); del buf540  # reuse
    # Source Nodes: [hidden_states_140], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg332_1, reinterpret_tensor(buf542, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg331_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf549)
    del arg331_1
    del arg332_1
    buf550 = buf532; del buf532  # reuse
    buf551 = buf531; del buf531  # reuse
    buf553 = reinterpret_tensor(buf542, (1, 512, 1024), (524288, 1024, 1), 0); del buf542  # reuse
    cpp_fused_add_native_layer_norm_82(c_void_p(buf530.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf553.data_ptr()))
    del arg333_1
    del arg334_1
    buf554 = reinterpret_tensor(buf528, (512, 4096), (4096, 1), 0); del buf528  # reuse
    # Source Nodes: [hidden_states_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg336_1, reinterpret_tensor(buf553, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg335_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf554)
    del arg335_1
    del arg336_1
    buf555 = reinterpret_tensor(buf554, (1, 512, 4096), (2097152, 4096, 1), 0); del buf554  # reuse
    cpp_fused_gelu_83(c_void_p(buf555.data_ptr()))
    buf556 = reinterpret_tensor(buf553, (512, 1024), (1024, 1), 0); del buf553  # reuse
    # Source Nodes: [hidden_states_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg338_1, reinterpret_tensor(buf555, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg337_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf556)
    del arg337_1
    del arg338_1
    buf557 = buf551; del buf551  # reuse
    buf558 = buf550; del buf550  # reuse
    buf560 = reinterpret_tensor(buf539, (1, 512, 1024), (524288, 1024, 1), 0); del buf539  # reuse
    cpp_fused_add_native_layer_norm_84(c_void_p(buf530.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf560.data_ptr()))
    del arg339_1
    del arg340_1
    buf561 = reinterpret_tensor(buf538, (512, 1024), (1024, 1), 0); del buf538  # reuse
    # Source Nodes: [mixed_query_layer_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg342_1, reinterpret_tensor(buf560, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg341_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf561)
    del arg341_1
    del arg342_1
    buf562 = reinterpret_tensor(buf534, (512, 1024), (1024, 1), 0); del buf534  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_21_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg344_1, reinterpret_tensor(buf560, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg343_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf562)
    del arg343_1
    del arg344_1
    buf563 = reinterpret_tensor(buf477, (512, 1024), (1024, 1), 0); del buf477  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_21_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg346_1, reinterpret_tensor(buf560, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg345_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf563)
    del arg345_1
    del arg346_1
    del buf560
    buf564 = reinterpret_tensor(buf561, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf561  # reuse
    buf565 = reinterpret_tensor(buf562, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf562  # reuse
    buf566 = reinterpret_tensor(buf563, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf563  # reuse
    cpp_fused_85(c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf567 = aten._scaled_dot_product_flash_attention(buf564, buf565, buf566, scale=0.125)
    del buf564
    buf568 = buf567[0]
    del buf567
    buf575 = reinterpret_tensor(buf566, (512, 1024), (1024, 1), 0); del buf566  # reuse
    # Source Nodes: [hidden_states_147], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg348_1, reinterpret_tensor(buf568, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg347_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf575)
    del arg347_1
    del arg348_1
    buf576 = buf558; del buf558  # reuse
    buf577 = buf557; del buf557  # reuse
    buf579 = reinterpret_tensor(buf568, (1, 512, 1024), (524288, 1024, 1), 0); del buf568  # reuse
    cpp_fused_add_native_layer_norm_86(c_void_p(buf530.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf579.data_ptr()))
    del arg349_1
    del arg350_1
    buf580 = reinterpret_tensor(buf555, (512, 4096), (4096, 1), 0); del buf555  # reuse
    # Source Nodes: [hidden_states_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg352_1, reinterpret_tensor(buf579, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg351_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf580)
    del arg351_1
    del arg352_1
    buf581 = reinterpret_tensor(buf580, (1, 512, 4096), (2097152, 4096, 1), 0); del buf580  # reuse
    cpp_fused_gelu_87(c_void_p(buf581.data_ptr()))
    buf582 = reinterpret_tensor(buf579, (512, 1024), (1024, 1), 0); del buf579  # reuse
    # Source Nodes: [hidden_states_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg354_1, reinterpret_tensor(buf581, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg353_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf582)
    del arg353_1
    del arg354_1
    buf583 = reinterpret_tensor(buf582, (1, 512, 1024), (524288, 1024, 1), 0); del buf582  # reuse
    buf584 = buf577; del buf577  # reuse
    buf585 = buf576; del buf576  # reuse
    buf587 = reinterpret_tensor(buf565, (1, 512, 1024), (524288, 1024, 1), 0); del buf565  # reuse
    cpp_fused_add_native_layer_norm_88(c_void_p(buf583.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf587.data_ptr()))
    del arg355_1
    del arg356_1
    buf588 = buf575; del buf575  # reuse
    # Source Nodes: [mixed_query_layer_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg358_1, reinterpret_tensor(buf587, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg357_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf588)
    del arg357_1
    del arg358_1
    buf589 = buf556; del buf556  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_22_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg360_1, reinterpret_tensor(buf587, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg359_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf589)
    del arg359_1
    del arg360_1
    buf590 = buf549; del buf549  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_22_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg362_1, reinterpret_tensor(buf587, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg361_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf590)
    del arg361_1
    del arg362_1
    buf591 = reinterpret_tensor(buf588, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf588  # reuse
    buf592 = reinterpret_tensor(buf589, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf589  # reuse
    buf593 = reinterpret_tensor(buf590, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf590  # reuse
    cpp_fused_89(c_void_p(buf591.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf593.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf594 = aten._scaled_dot_product_flash_attention(buf591, buf592, buf593, scale=0.125)
    buf595 = buf594[0]
    del buf594
    buf602 = reinterpret_tensor(buf593, (512, 1024), (1024, 1), 0); del buf593  # reuse
    # Source Nodes: [hidden_states_154], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg364_1, reinterpret_tensor(buf595, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg363_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf602)
    del arg363_1
    del arg364_1
    buf603 = buf585; del buf585  # reuse
    buf604 = buf584; del buf584  # reuse
    buf606 = reinterpret_tensor(buf595, (1, 512, 1024), (524288, 1024, 1), 0); del buf595  # reuse
    cpp_fused_add_native_layer_norm_90(c_void_p(buf583.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf606.data_ptr()))
    del arg365_1
    del arg366_1
    buf607 = reinterpret_tensor(buf581, (512, 4096), (4096, 1), 0); del buf581  # reuse
    # Source Nodes: [hidden_states_156], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg368_1, reinterpret_tensor(buf606, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg367_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf607)
    del arg367_1
    del arg368_1
    buf608 = reinterpret_tensor(buf607, (1, 512, 4096), (2097152, 4096, 1), 0); del buf607  # reuse
    cpp_fused_gelu_91(c_void_p(buf608.data_ptr()))
    buf609 = reinterpret_tensor(buf606, (512, 1024), (1024, 1), 0); del buf606  # reuse
    # Source Nodes: [hidden_states_158], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg370_1, reinterpret_tensor(buf608, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg369_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf609)
    del arg369_1
    del arg370_1
    buf610 = buf604; del buf604  # reuse
    buf611 = buf603; del buf603  # reuse
    buf613 = reinterpret_tensor(buf592, (1, 512, 1024), (524288, 1024, 1), 0); del buf592  # reuse
    cpp_fused_add_native_layer_norm_92(c_void_p(buf583.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf613.data_ptr()))
    del arg371_1
    del arg372_1
    buf614 = reinterpret_tensor(buf591, (512, 1024), (1024, 1), 0); del buf591  # reuse
    # Source Nodes: [mixed_query_layer_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg374_1, reinterpret_tensor(buf613, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg373_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf614)
    del arg373_1
    del arg374_1
    buf615 = reinterpret_tensor(buf587, (512, 1024), (1024, 1), 0); del buf587  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_23_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg376_1, reinterpret_tensor(buf613, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg375_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf615)
    del arg375_1
    del arg376_1
    buf616 = reinterpret_tensor(buf530, (512, 1024), (1024, 1), 0); del buf530  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_23_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg378_1, reinterpret_tensor(buf613, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg377_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf616)
    del arg377_1
    del arg378_1
    del buf613
    buf617 = reinterpret_tensor(buf614, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf614  # reuse
    buf618 = reinterpret_tensor(buf615, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf615  # reuse
    buf619 = reinterpret_tensor(buf616, (1, 16, 512, 64), (524288, 64, 1024, 1), 0); del buf616  # reuse
    cpp_fused_93(c_void_p(buf617.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf619.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf620 = aten._scaled_dot_product_flash_attention(buf617, buf618, buf619, scale=0.125)
    del buf617
    buf621 = buf620[0]
    del buf620
    buf628 = reinterpret_tensor(buf619, (512, 1024), (1024, 1), 0); del buf619  # reuse
    # Source Nodes: [hidden_states_161], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg380_1, reinterpret_tensor(buf621, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg379_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf628)
    del arg379_1
    del arg380_1
    buf629 = buf611; del buf611  # reuse
    buf630 = buf610; del buf610  # reuse
    buf632 = reinterpret_tensor(buf621, (1, 512, 1024), (524288, 1024, 1), 0); del buf621  # reuse
    cpp_fused_add_native_layer_norm_94(c_void_p(buf583.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(arg381_1.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf632.data_ptr()))
    del arg381_1
    del arg382_1
    buf633 = reinterpret_tensor(buf608, (512, 4096), (4096, 1), 0); del buf608  # reuse
    # Source Nodes: [hidden_states_163], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg384_1, reinterpret_tensor(buf632, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg383_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf633)
    del arg383_1
    del arg384_1
    buf634 = reinterpret_tensor(buf633, (1, 512, 4096), (2097152, 4096, 1), 0); del buf633  # reuse
    cpp_fused_gelu_95(c_void_p(buf634.data_ptr()))
    buf635 = reinterpret_tensor(buf632, (512, 1024), (1024, 1), 0); del buf632  # reuse
    # Source Nodes: [hidden_states_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg386_1, reinterpret_tensor(buf634, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg385_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf635)
    del arg385_1
    del arg386_1
    del buf634
    buf636 = reinterpret_tensor(buf635, (1, 512, 1024), (524288, 1024, 1), 0); del buf635  # reuse
    buf637 = buf630; del buf630  # reuse
    buf638 = buf629; del buf629  # reuse
    buf640 = reinterpret_tensor(buf618, (1, 512, 1024), (524288, 1024, 1), 0); del buf618  # reuse
    cpp_fused_add_native_layer_norm_96(c_void_p(buf636.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(arg387_1.data_ptr()), c_void_p(arg388_1.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf640.data_ptr()))
    del arg387_1
    del arg388_1
    del buf583
    del buf602
    del buf609
    del buf628
    buf641 = reinterpret_tensor(buf636, (512, 1024), (1024, 1), 0); del buf636  # reuse
    # Source Nodes: [hidden_states_169], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg390_1, reinterpret_tensor(buf640, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg389_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf641)
    del arg389_1
    del arg390_1
    buf642 = buf638; del buf638  # reuse
    buf643 = buf637; del buf637  # reuse
    buf645 = buf640; del buf640  # reuse
    cpp_fused_gelu_native_layer_norm_97(c_void_p(buf641.data_ptr()), c_void_p(arg391_1.data_ptr()), c_void_p(arg392_1.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf643.data_ptr()), c_void_p(buf645.data_ptr()))
    del arg391_1
    del arg392_1
    del buf641
    del buf642
    del buf643
    buf646 = empty((512, 29056), device='cpu', dtype=torch.float32)
    # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg394_1, reinterpret_tensor(buf645, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg393_1, (1024, 29056), (1, 1024), 0), alpha=1, beta=1, out=buf646)
    del arg393_1
    del arg394_1
    del buf645
    buf647 = empty_strided((511, 1), (1, 511), device='cpu', dtype=torch.float32)
    buf648 = empty_strided((511, 1), (1, 511), device='cpu', dtype=torch.float32)
    buf649 = empty((), device='cpu', dtype=torch.float32)
    buf650 = empty((), device='cpu', dtype=torch.int64)
    buf651 = buf649; del buf649  # reuse
    cpp_fused__log_softmax_nll_loss_forward_98(c_void_p(buf651.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf650.data_ptr()))
    del arg396_1
    return (buf651, reinterpret_tensor(buf646, (1, 512, 29056), (14876672, 29056, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((29056, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((2, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((29056, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((29056, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg396_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg397_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MegatronBertForCausalLM', benchmark_compiled_module)
