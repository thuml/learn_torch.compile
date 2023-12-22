
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


cpp_fused_add_embedding_native_layer_norm_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
                       const float* in_ptr3,
                       const long* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = in_ptr2[static_cast<long>(x0)];
                    auto tmp11 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 30000);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 30000L), "index out of bounds: 0 <= tmp3 < 30000L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*tmp3)));
                    auto tmp6 = decltype(tmp5)(tmp5 + 2);
                    auto tmp7 = tmp5 < 0;
                    auto tmp8 = tmp7 ? tmp6 : tmp5;
                    TORCH_CHECK((0 <= tmp8) & (tmp8 < 2L), "index out of bounds: 0 <= tmp8 < 2L")
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*tmp8)));
                    auto tmp10 = tmp4 + tmp9;
                    auto tmp12 = decltype(tmp11)(tmp11 + 512);
                    auto tmp13 = tmp11 < 0;
                    auto tmp14 = tmp13 ? tmp12 : tmp11;
                    TORCH_CHECK((0 <= tmp14) & (tmp14 < 512L), "index out of bounds: 0 <= tmp14 < 512L")
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (128L*tmp14)));
                    auto tmp16 = tmp10 + tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp16);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp4 = out_ptr2[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-12);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_8 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_16 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_24 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_32 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_40 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_48 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(4096.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_pow_tanh_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = static_cast<float>(0.5);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp4 = tmp0 * tmp0;
                    auto tmp5 = tmp4 * tmp0;
                    auto tmp6 = static_cast<float>(0.044715);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.7978845608028654);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                    auto tmp14 = static_cast<float>(1.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 + tmp15;
                    auto tmp17 = tmp3 * tmp16;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp18 = out_ptr0[static_cast<long>(x0)];
                auto tmp21 = out_ptr1[static_cast<long>(x0)];
                auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = at::vec::Vectorized<float>(tmp18);
                auto tmp20 = tmp17 - tmp19;
                auto tmp22 = static_cast<float>(128.0);
                auto tmp23 = tmp21 / tmp22;
                auto tmp24 = static_cast<float>(1e-12);
                auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                auto tmp26 = 1 / std::sqrt(tmp25);
                auto tmp27 = at::vec::Vectorized<float>(tmp26);
                auto tmp28 = tmp20 * tmp27;
                auto tmp30 = tmp28 * tmp29;
                auto tmp32 = tmp30 + tmp31;
                tmp32.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused__log_softmax_nll_loss_forward_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(30000L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (30000L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(30000L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (30000L*x0)));
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
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x0)];
                        auto tmp9 = out_ptr0[static_cast<long>(x0)];
                        auto tmp11 = out_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = static_cast<long>(0);
                        auto tmp4 = tmp2 ? tmp0 : tmp3;
                        auto tmp5 = decltype(tmp4)(tmp4 + 30000);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 30000L), "index out of bounds: 0 <= tmp7 < 30000L")
                        auto tmp8 = in_ptr0[static_cast<long>(tmp7 + (30000L*x0))];
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1 = args
    args.clear()
    assert_size_stride(arg0_1, (30000, 128), (128, 1))
    assert_size_stride(arg1_1, (2, 128), (128, 1))
    assert_size_stride(arg2_1, (512, 128), (128, 1))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (4096, 128), (128, 1))
    assert_size_stride(arg6_1, (4096, ), (1, ))
    assert_size_stride(arg7_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg8_1, (4096, ), (1, ))
    assert_size_stride(arg9_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg10_1, (4096, ), (1, ))
    assert_size_stride(arg11_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg12_1, (4096, ), (1, ))
    assert_size_stride(arg13_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg14_1, (4096, ), (1, ))
    assert_size_stride(arg15_1, (4096, ), (1, ))
    assert_size_stride(arg16_1, (4096, ), (1, ))
    assert_size_stride(arg17_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg18_1, (16384, ), (1, ))
    assert_size_stride(arg19_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg20_1, (4096, ), (1, ))
    assert_size_stride(arg21_1, (4096, ), (1, ))
    assert_size_stride(arg22_1, (4096, ), (1, ))
    assert_size_stride(arg23_1, (128, 4096), (4096, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (30000, 128), (128, 1))
    assert_size_stride(arg28_1, (30000, ), (1, ))
    assert_size_stride(arg29_1, (1, 512), (512, 1))
    assert_size_stride(arg30_1, (1, 512), (512, 1))
    assert_size_stride(arg31_1, (1, 512), (512, 1))
    assert_size_stride(arg32_1, (1, 512), (512, 1))
    buf0 = empty((1, 512, 128), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf4 = empty((1, 512, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_0(c_void_p(arg31_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg0_1
    del arg1_1
    del arg29_1
    del arg2_1
    del arg30_1
    del arg31_1
    del arg3_1
    del arg4_1
    buf5 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg6_1, reinterpret_tensor(buf4, (512, 128), (128, 1), 0), reinterpret_tensor(arg5_1, (128, 4096), (1, 128), 0), alpha=1, beta=1, out=buf5)
    del arg5_1
    del arg6_1
    buf6 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf5, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf6)
    buf7 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_key_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf5, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf7)
    buf8 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_value_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf5, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf8)
    buf9 = reinterpret_tensor(buf6, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf6  # reuse
    buf10 = reinterpret_tensor(buf7, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf7  # reuse
    buf11 = reinterpret_tensor(buf8, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf8  # reuse
    cpp_fused_1(c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf12 = aten._scaled_dot_product_flash_attention(buf9, buf10, buf11, scale=0.125)
    del buf10
    buf13 = buf12[0]
    del buf12
    buf20 = reinterpret_tensor(buf9, (512, 4096), (4096, 1), 0); del buf9  # reuse
    # Source Nodes: [projected_context_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf13, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf20)
    buf21 = buf2; del buf2  # reuse
    buf22 = buf1; del buf1  # reuse
    buf24 = reinterpret_tensor(buf13, (1, 512, 4096), (2097152, 4096, 1), 0); del buf13  # reuse
    cpp_fused_add_native_layer_norm_2(c_void_p(buf5.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()))
    buf25 = empty((512, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [ffn_output], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf24, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf25)
    buf26 = reinterpret_tensor(buf25, (1, 512, 16384), (8388608, 16384, 1), 0); del buf25  # reuse
    cpp_fused_add_mul_pow_tanh_3(c_void_p(buf26.data_ptr()))
    buf27 = buf5; del buf5  # reuse
    # Source Nodes: [ffn_output_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf26, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf27)
    buf28 = buf22; del buf22  # reuse
    buf29 = buf21; del buf21  # reuse
    buf31 = reinterpret_tensor(buf20, (1, 512, 4096), (2097152, 4096, 1), 0); del buf20  # reuse
    cpp_fused_add_native_layer_norm_4(c_void_p(buf27.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()))
    buf32 = buf27; del buf27  # reuse
    # Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf31, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf32)
    buf33 = reinterpret_tensor(buf24, (512, 4096), (4096, 1), 0); del buf24  # reuse
    # Source Nodes: [mixed_key_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf31, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf33)
    buf34 = reinterpret_tensor(buf11, (512, 4096), (4096, 1), 0); del buf11  # reuse
    # Source Nodes: [mixed_value_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf31, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf34)
    buf35 = reinterpret_tensor(buf32, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf32  # reuse
    buf36 = reinterpret_tensor(buf33, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf33  # reuse
    buf37 = reinterpret_tensor(buf34, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf34  # reuse
    cpp_fused_5(c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf38 = aten._scaled_dot_product_flash_attention(buf35, buf36, buf37, scale=0.125)
    del buf35
    buf39 = buf38[0]
    del buf38
    buf46 = reinterpret_tensor(buf37, (512, 4096), (4096, 1), 0); del buf37  # reuse
    # Source Nodes: [projected_context_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf39, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf46)
    buf47 = buf29; del buf29  # reuse
    buf48 = buf28; del buf28  # reuse
    buf50 = reinterpret_tensor(buf39, (1, 512, 4096), (2097152, 4096, 1), 0); del buf39  # reuse
    cpp_fused_add_native_layer_norm_6(c_void_p(buf31.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()))
    buf51 = reinterpret_tensor(buf26, (512, 16384), (16384, 1), 0); del buf26  # reuse
    # Source Nodes: [ffn_output_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf50, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf51)
    buf52 = reinterpret_tensor(buf51, (1, 512, 16384), (8388608, 16384, 1), 0); del buf51  # reuse
    cpp_fused_add_mul_pow_tanh_7(c_void_p(buf52.data_ptr()))
    buf53 = buf46; del buf46  # reuse
    # Source Nodes: [ffn_output_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf52, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf53)
    buf54 = buf48; del buf48  # reuse
    buf55 = buf47; del buf47  # reuse
    buf57 = buf31; del buf31  # reuse
    cpp_fused_add_native_layer_norm_8(c_void_p(buf53.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()))
    buf58 = buf53; del buf53  # reuse
    # Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf57, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf58)
    buf59 = reinterpret_tensor(buf50, (512, 4096), (4096, 1), 0); del buf50  # reuse
    # Source Nodes: [mixed_key_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf57, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf59)
    buf60 = reinterpret_tensor(buf36, (512, 4096), (4096, 1), 0); del buf36  # reuse
    # Source Nodes: [mixed_value_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf57, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf60)
    buf61 = reinterpret_tensor(buf58, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf58  # reuse
    buf62 = reinterpret_tensor(buf59, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf59  # reuse
    buf63 = reinterpret_tensor(buf60, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf60  # reuse
    cpp_fused_9(c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf64 = aten._scaled_dot_product_flash_attention(buf61, buf62, buf63, scale=0.125)
    del buf61
    buf65 = buf64[0]
    del buf64
    buf72 = reinterpret_tensor(buf63, (512, 4096), (4096, 1), 0); del buf63  # reuse
    # Source Nodes: [projected_context_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf65, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf72)
    buf73 = buf55; del buf55  # reuse
    buf74 = buf54; del buf54  # reuse
    buf76 = reinterpret_tensor(buf65, (1, 512, 4096), (2097152, 4096, 1), 0); del buf65  # reuse
    cpp_fused_add_native_layer_norm_10(c_void_p(buf57.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf76.data_ptr()))
    buf77 = reinterpret_tensor(buf52, (512, 16384), (16384, 1), 0); del buf52  # reuse
    # Source Nodes: [ffn_output_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf76, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf77)
    buf78 = reinterpret_tensor(buf77, (1, 512, 16384), (8388608, 16384, 1), 0); del buf77  # reuse
    cpp_fused_add_mul_pow_tanh_11(c_void_p(buf78.data_ptr()))
    buf79 = buf72; del buf72  # reuse
    # Source Nodes: [ffn_output_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf78, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf79)
    buf80 = buf74; del buf74  # reuse
    buf81 = buf73; del buf73  # reuse
    buf83 = buf57; del buf57  # reuse
    cpp_fused_add_native_layer_norm_12(c_void_p(buf79.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()))
    buf84 = buf79; del buf79  # reuse
    # Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf83, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf84)
    buf85 = reinterpret_tensor(buf76, (512, 4096), (4096, 1), 0); del buf76  # reuse
    # Source Nodes: [mixed_key_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf83, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf85)
    buf86 = reinterpret_tensor(buf62, (512, 4096), (4096, 1), 0); del buf62  # reuse
    # Source Nodes: [mixed_value_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf83, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf86)
    buf87 = reinterpret_tensor(buf84, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf84  # reuse
    buf88 = reinterpret_tensor(buf85, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf85  # reuse
    buf89 = reinterpret_tensor(buf86, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf86  # reuse
    cpp_fused_13(c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf90 = aten._scaled_dot_product_flash_attention(buf87, buf88, buf89, scale=0.125)
    del buf87
    buf91 = buf90[0]
    del buf90
    buf98 = reinterpret_tensor(buf89, (512, 4096), (4096, 1), 0); del buf89  # reuse
    # Source Nodes: [projected_context_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf91, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf98)
    buf99 = buf81; del buf81  # reuse
    buf100 = buf80; del buf80  # reuse
    buf102 = reinterpret_tensor(buf91, (1, 512, 4096), (2097152, 4096, 1), 0); del buf91  # reuse
    cpp_fused_add_native_layer_norm_14(c_void_p(buf83.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()))
    buf103 = reinterpret_tensor(buf78, (512, 16384), (16384, 1), 0); del buf78  # reuse
    # Source Nodes: [ffn_output_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf102, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf103)
    buf104 = reinterpret_tensor(buf103, (1, 512, 16384), (8388608, 16384, 1), 0); del buf103  # reuse
    cpp_fused_add_mul_pow_tanh_15(c_void_p(buf104.data_ptr()))
    buf105 = buf98; del buf98  # reuse
    # Source Nodes: [ffn_output_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf104, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf105)
    buf106 = buf99; del buf99  # reuse
    buf107 = buf100; del buf100  # reuse
    buf109 = buf83; del buf83  # reuse
    cpp_fused_add_native_layer_norm_16(c_void_p(buf105.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()))
    buf110 = buf105; del buf105  # reuse
    # Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf109, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf110)
    buf111 = reinterpret_tensor(buf102, (512, 4096), (4096, 1), 0); del buf102  # reuse
    # Source Nodes: [mixed_key_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf109, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf111)
    buf112 = reinterpret_tensor(buf88, (512, 4096), (4096, 1), 0); del buf88  # reuse
    # Source Nodes: [mixed_value_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf109, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf112)
    buf113 = reinterpret_tensor(buf110, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf110  # reuse
    buf114 = reinterpret_tensor(buf111, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf111  # reuse
    buf115 = reinterpret_tensor(buf112, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf112  # reuse
    cpp_fused_17(c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf116 = aten._scaled_dot_product_flash_attention(buf113, buf114, buf115, scale=0.125)
    del buf113
    buf117 = buf116[0]
    del buf116
    buf124 = reinterpret_tensor(buf115, (512, 4096), (4096, 1), 0); del buf115  # reuse
    # Source Nodes: [projected_context_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf117, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf124)
    buf125 = buf107; del buf107  # reuse
    buf126 = buf106; del buf106  # reuse
    buf128 = reinterpret_tensor(buf117, (1, 512, 4096), (2097152, 4096, 1), 0); del buf117  # reuse
    cpp_fused_add_native_layer_norm_18(c_void_p(buf109.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()))
    buf129 = reinterpret_tensor(buf104, (512, 16384), (16384, 1), 0); del buf104  # reuse
    # Source Nodes: [ffn_output_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf128, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf129)
    buf130 = reinterpret_tensor(buf129, (1, 512, 16384), (8388608, 16384, 1), 0); del buf129  # reuse
    cpp_fused_add_mul_pow_tanh_19(c_void_p(buf130.data_ptr()))
    buf131 = buf124; del buf124  # reuse
    # Source Nodes: [ffn_output_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf130, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf131)
    buf132 = buf126; del buf126  # reuse
    buf133 = buf125; del buf125  # reuse
    buf135 = buf109; del buf109  # reuse
    cpp_fused_add_native_layer_norm_20(c_void_p(buf131.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()))
    buf136 = buf131; del buf131  # reuse
    # Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf135, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf136)
    buf137 = reinterpret_tensor(buf128, (512, 4096), (4096, 1), 0); del buf128  # reuse
    # Source Nodes: [mixed_key_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf135, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf137)
    buf138 = reinterpret_tensor(buf114, (512, 4096), (4096, 1), 0); del buf114  # reuse
    # Source Nodes: [mixed_value_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf135, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf138)
    buf139 = reinterpret_tensor(buf136, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf136  # reuse
    buf140 = reinterpret_tensor(buf137, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf137  # reuse
    buf141 = reinterpret_tensor(buf138, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf138  # reuse
    cpp_fused_21(c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf142 = aten._scaled_dot_product_flash_attention(buf139, buf140, buf141, scale=0.125)
    del buf139
    buf143 = buf142[0]
    del buf142
    buf150 = reinterpret_tensor(buf141, (512, 4096), (4096, 1), 0); del buf141  # reuse
    # Source Nodes: [projected_context_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf143, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf150)
    buf151 = buf133; del buf133  # reuse
    buf152 = buf132; del buf132  # reuse
    buf154 = reinterpret_tensor(buf143, (1, 512, 4096), (2097152, 4096, 1), 0); del buf143  # reuse
    cpp_fused_add_native_layer_norm_22(c_void_p(buf135.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()))
    buf155 = reinterpret_tensor(buf130, (512, 16384), (16384, 1), 0); del buf130  # reuse
    # Source Nodes: [ffn_output_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf154, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf155)
    buf156 = reinterpret_tensor(buf155, (1, 512, 16384), (8388608, 16384, 1), 0); del buf155  # reuse
    cpp_fused_add_mul_pow_tanh_23(c_void_p(buf156.data_ptr()))
    buf157 = buf150; del buf150  # reuse
    # Source Nodes: [ffn_output_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf156, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf157)
    buf158 = buf152; del buf152  # reuse
    buf159 = buf151; del buf151  # reuse
    buf161 = buf135; del buf135  # reuse
    cpp_fused_add_native_layer_norm_24(c_void_p(buf157.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()))
    buf162 = buf157; del buf157  # reuse
    # Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf161, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf162)
    buf163 = reinterpret_tensor(buf154, (512, 4096), (4096, 1), 0); del buf154  # reuse
    # Source Nodes: [mixed_key_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf161, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf163)
    buf164 = reinterpret_tensor(buf140, (512, 4096), (4096, 1), 0); del buf140  # reuse
    # Source Nodes: [mixed_value_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf161, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf164)
    buf165 = reinterpret_tensor(buf162, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf162  # reuse
    buf166 = reinterpret_tensor(buf163, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf163  # reuse
    buf167 = reinterpret_tensor(buf164, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf164  # reuse
    cpp_fused_25(c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf168 = aten._scaled_dot_product_flash_attention(buf165, buf166, buf167, scale=0.125)
    del buf165
    buf169 = buf168[0]
    del buf168
    buf176 = reinterpret_tensor(buf167, (512, 4096), (4096, 1), 0); del buf167  # reuse
    # Source Nodes: [projected_context_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf169, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf176)
    buf177 = buf159; del buf159  # reuse
    buf178 = buf158; del buf158  # reuse
    buf180 = reinterpret_tensor(buf169, (1, 512, 4096), (2097152, 4096, 1), 0); del buf169  # reuse
    cpp_fused_add_native_layer_norm_26(c_void_p(buf161.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()))
    buf181 = reinterpret_tensor(buf156, (512, 16384), (16384, 1), 0); del buf156  # reuse
    # Source Nodes: [ffn_output_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf180, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf181)
    buf182 = reinterpret_tensor(buf181, (1, 512, 16384), (8388608, 16384, 1), 0); del buf181  # reuse
    cpp_fused_add_mul_pow_tanh_27(c_void_p(buf182.data_ptr()))
    buf183 = buf176; del buf176  # reuse
    # Source Nodes: [ffn_output_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf182, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf183)
    buf184 = buf178; del buf178  # reuse
    buf185 = buf177; del buf177  # reuse
    buf187 = buf161; del buf161  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf183.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()))
    buf188 = buf183; del buf183  # reuse
    # Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf187, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf188)
    buf189 = reinterpret_tensor(buf180, (512, 4096), (4096, 1), 0); del buf180  # reuse
    # Source Nodes: [mixed_key_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf187, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf189)
    buf190 = reinterpret_tensor(buf166, (512, 4096), (4096, 1), 0); del buf166  # reuse
    # Source Nodes: [mixed_value_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf187, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf190)
    buf191 = reinterpret_tensor(buf188, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf188  # reuse
    buf192 = reinterpret_tensor(buf189, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf189  # reuse
    buf193 = reinterpret_tensor(buf190, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf190  # reuse
    cpp_fused_29(c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf194 = aten._scaled_dot_product_flash_attention(buf191, buf192, buf193, scale=0.125)
    del buf191
    buf195 = buf194[0]
    del buf194
    buf202 = reinterpret_tensor(buf193, (512, 4096), (4096, 1), 0); del buf193  # reuse
    # Source Nodes: [projected_context_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf195, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf202)
    buf203 = buf185; del buf185  # reuse
    buf204 = buf184; del buf184  # reuse
    buf206 = reinterpret_tensor(buf195, (1, 512, 4096), (2097152, 4096, 1), 0); del buf195  # reuse
    cpp_fused_add_native_layer_norm_30(c_void_p(buf187.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf206.data_ptr()))
    buf207 = reinterpret_tensor(buf182, (512, 16384), (16384, 1), 0); del buf182  # reuse
    # Source Nodes: [ffn_output_28], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf206, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf207)
    buf208 = reinterpret_tensor(buf207, (1, 512, 16384), (8388608, 16384, 1), 0); del buf207  # reuse
    cpp_fused_add_mul_pow_tanh_31(c_void_p(buf208.data_ptr()))
    buf209 = buf202; del buf202  # reuse
    # Source Nodes: [ffn_output_31], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf208, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf209)
    buf210 = buf204; del buf204  # reuse
    buf211 = buf203; del buf203  # reuse
    buf213 = buf187; del buf187  # reuse
    cpp_fused_add_native_layer_norm_32(c_void_p(buf209.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf213.data_ptr()))
    buf214 = buf209; del buf209  # reuse
    # Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf213, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf214)
    buf215 = reinterpret_tensor(buf206, (512, 4096), (4096, 1), 0); del buf206  # reuse
    # Source Nodes: [mixed_key_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf213, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf215)
    buf216 = reinterpret_tensor(buf192, (512, 4096), (4096, 1), 0); del buf192  # reuse
    # Source Nodes: [mixed_value_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf213, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf216)
    buf217 = reinterpret_tensor(buf214, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf214  # reuse
    buf218 = reinterpret_tensor(buf215, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf215  # reuse
    buf219 = reinterpret_tensor(buf216, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf216  # reuse
    cpp_fused_33(c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf220 = aten._scaled_dot_product_flash_attention(buf217, buf218, buf219, scale=0.125)
    del buf217
    buf221 = buf220[0]
    del buf220
    buf228 = reinterpret_tensor(buf219, (512, 4096), (4096, 1), 0); del buf219  # reuse
    # Source Nodes: [projected_context_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf221, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf228)
    buf229 = buf211; del buf211  # reuse
    buf230 = buf210; del buf210  # reuse
    buf232 = reinterpret_tensor(buf221, (1, 512, 4096), (2097152, 4096, 1), 0); del buf221  # reuse
    cpp_fused_add_native_layer_norm_34(c_void_p(buf213.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()))
    buf233 = reinterpret_tensor(buf208, (512, 16384), (16384, 1), 0); del buf208  # reuse
    # Source Nodes: [ffn_output_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf232, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf233)
    buf234 = reinterpret_tensor(buf233, (1, 512, 16384), (8388608, 16384, 1), 0); del buf233  # reuse
    cpp_fused_add_mul_pow_tanh_35(c_void_p(buf234.data_ptr()))
    buf235 = buf228; del buf228  # reuse
    # Source Nodes: [ffn_output_35], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf234, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf235)
    buf236 = buf230; del buf230  # reuse
    buf237 = buf229; del buf229  # reuse
    buf239 = buf213; del buf213  # reuse
    cpp_fused_add_native_layer_norm_36(c_void_p(buf235.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()))
    buf240 = buf235; del buf235  # reuse
    # Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf239, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf240)
    buf241 = reinterpret_tensor(buf232, (512, 4096), (4096, 1), 0); del buf232  # reuse
    # Source Nodes: [mixed_key_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf239, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf241)
    buf242 = reinterpret_tensor(buf218, (512, 4096), (4096, 1), 0); del buf218  # reuse
    # Source Nodes: [mixed_value_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf239, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf242)
    buf243 = reinterpret_tensor(buf240, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf240  # reuse
    buf244 = reinterpret_tensor(buf241, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf241  # reuse
    buf245 = reinterpret_tensor(buf242, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf242  # reuse
    cpp_fused_37(c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf246 = aten._scaled_dot_product_flash_attention(buf243, buf244, buf245, scale=0.125)
    del buf243
    buf247 = buf246[0]
    del buf246
    buf254 = reinterpret_tensor(buf245, (512, 4096), (4096, 1), 0); del buf245  # reuse
    # Source Nodes: [projected_context_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf247, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf254)
    buf255 = buf237; del buf237  # reuse
    buf256 = buf236; del buf236  # reuse
    buf258 = reinterpret_tensor(buf247, (1, 512, 4096), (2097152, 4096, 1), 0); del buf247  # reuse
    cpp_fused_add_native_layer_norm_38(c_void_p(buf239.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()))
    buf259 = reinterpret_tensor(buf234, (512, 16384), (16384, 1), 0); del buf234  # reuse
    # Source Nodes: [ffn_output_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf258, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf259)
    buf260 = reinterpret_tensor(buf259, (1, 512, 16384), (8388608, 16384, 1), 0); del buf259  # reuse
    cpp_fused_add_mul_pow_tanh_39(c_void_p(buf260.data_ptr()))
    buf261 = buf254; del buf254  # reuse
    # Source Nodes: [ffn_output_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf260, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf261)
    buf262 = buf256; del buf256  # reuse
    buf263 = buf255; del buf255  # reuse
    buf265 = buf239; del buf239  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf261.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf265.data_ptr()))
    buf266 = buf261; del buf261  # reuse
    # Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf265, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf266)
    buf267 = reinterpret_tensor(buf258, (512, 4096), (4096, 1), 0); del buf258  # reuse
    # Source Nodes: [mixed_key_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf265, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf267)
    buf268 = reinterpret_tensor(buf244, (512, 4096), (4096, 1), 0); del buf244  # reuse
    # Source Nodes: [mixed_value_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf265, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf268)
    buf269 = reinterpret_tensor(buf266, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf266  # reuse
    buf270 = reinterpret_tensor(buf267, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf267  # reuse
    buf271 = reinterpret_tensor(buf268, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf268  # reuse
    cpp_fused_41(c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf272 = aten._scaled_dot_product_flash_attention(buf269, buf270, buf271, scale=0.125)
    del buf269
    buf273 = buf272[0]
    del buf272
    buf280 = reinterpret_tensor(buf271, (512, 4096), (4096, 1), 0); del buf271  # reuse
    # Source Nodes: [projected_context_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf273, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf280)
    buf281 = buf263; del buf263  # reuse
    buf282 = buf262; del buf262  # reuse
    buf284 = reinterpret_tensor(buf273, (1, 512, 4096), (2097152, 4096, 1), 0); del buf273  # reuse
    cpp_fused_add_native_layer_norm_42(c_void_p(buf265.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()))
    buf285 = reinterpret_tensor(buf260, (512, 16384), (16384, 1), 0); del buf260  # reuse
    # Source Nodes: [ffn_output_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf284, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf285)
    buf286 = reinterpret_tensor(buf285, (1, 512, 16384), (8388608, 16384, 1), 0); del buf285  # reuse
    cpp_fused_add_mul_pow_tanh_43(c_void_p(buf286.data_ptr()))
    buf287 = buf280; del buf280  # reuse
    # Source Nodes: [ffn_output_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf286, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf287)
    buf288 = buf282; del buf282  # reuse
    buf289 = buf281; del buf281  # reuse
    buf291 = buf265; del buf265  # reuse
    cpp_fused_add_native_layer_norm_44(c_void_p(buf287.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()))
    buf292 = buf287; del buf287  # reuse
    # Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf291, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf292)
    del arg7_1
    del arg8_1
    buf293 = reinterpret_tensor(buf284, (512, 4096), (4096, 1), 0); del buf284  # reuse
    # Source Nodes: [mixed_key_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf291, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf293)
    del arg10_1
    del arg9_1
    buf294 = reinterpret_tensor(buf270, (512, 4096), (4096, 1), 0); del buf270  # reuse
    # Source Nodes: [mixed_value_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf291, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg11_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf294)
    del arg11_1
    del arg12_1
    buf295 = reinterpret_tensor(buf292, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf292  # reuse
    buf296 = reinterpret_tensor(buf293, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf293  # reuse
    buf297 = reinterpret_tensor(buf294, (1, 64, 512, 64), (2097152, 64, 4096, 1), 0); del buf294  # reuse
    cpp_fused_45(c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf298 = aten._scaled_dot_product_flash_attention(buf295, buf296, buf297, scale=0.125)
    del buf295
    del buf296
    buf299 = buf298[0]
    del buf298
    buf306 = reinterpret_tensor(buf297, (512, 4096), (4096, 1), 0); del buf297  # reuse
    # Source Nodes: [projected_context_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf299, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf306)
    del arg13_1
    del arg14_1
    buf307 = buf289; del buf289  # reuse
    buf308 = buf288; del buf288  # reuse
    buf310 = reinterpret_tensor(buf299, (1, 512, 4096), (2097152, 4096, 1), 0); del buf299  # reuse
    cpp_fused_add_native_layer_norm_46(c_void_p(buf291.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf310.data_ptr()))
    del arg15_1
    del arg16_1
    buf311 = reinterpret_tensor(buf286, (512, 16384), (16384, 1), 0); del buf286  # reuse
    # Source Nodes: [ffn_output_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf310, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf311)
    del arg17_1
    del arg18_1
    buf312 = reinterpret_tensor(buf311, (1, 512, 16384), (8388608, 16384, 1), 0); del buf311  # reuse
    cpp_fused_add_mul_pow_tanh_47(c_void_p(buf312.data_ptr()))
    buf313 = buf306; del buf306  # reuse
    # Source Nodes: [ffn_output_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf312, (512, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf313)
    del arg19_1
    del arg20_1
    del buf312
    buf314 = buf308; del buf308  # reuse
    buf315 = buf307; del buf307  # reuse
    buf317 = buf291; del buf291  # reuse
    cpp_fused_add_native_layer_norm_48(c_void_p(buf313.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf317.data_ptr()))
    del arg21_1
    del arg22_1
    del buf310
    del buf313
    buf318 = reinterpret_tensor(buf4, (512, 128), (128, 1), 0); del buf4  # reuse
    # Source Nodes: [hidden_states_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg24_1, reinterpret_tensor(buf317, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg23_1, (4096, 128), (1, 4096), 0), alpha=1, beta=1, out=buf318)
    del arg23_1
    del arg24_1
    del buf317
    buf319 = buf315; del buf315  # reuse
    buf320 = buf314; del buf314  # reuse
    buf322 = buf0; del buf0  # reuse
    cpp_fused_add_mul_native_layer_norm_pow_tanh_49(c_void_p(buf318.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf322.data_ptr()))
    del arg25_1
    del arg26_1
    del buf318
    buf323 = empty((512, 30000), device='cpu', dtype=torch.float32)
    # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg28_1, reinterpret_tensor(buf322, (512, 128), (128, 1), 0), reinterpret_tensor(arg27_1, (128, 30000), (1, 128), 0), alpha=1, beta=1, out=buf323)
    del arg27_1
    del arg28_1
    del buf322
    buf324 = reinterpret_tensor(buf320, (512, 1), (1, 512), 0); del buf320  # reuse
    buf325 = reinterpret_tensor(buf319, (512, 1), (1, 512), 0); del buf319  # reuse
    buf326 = empty((), device='cpu', dtype=torch.float32)
    buf327 = empty((), device='cpu', dtype=torch.int64)
    buf328 = buf326; del buf326  # reuse
    cpp_fused__log_softmax_nll_loss_forward_50(c_void_p(buf328.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf327.data_ptr()))
    del arg32_1
    return (buf328, reinterpret_tensor(buf323, (1, 512, 30000), (15360000, 30000, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30000, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((2, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((4096, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((128, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((30000, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((30000, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg30_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg31_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg32_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AlbertForMaskedLM', benchmark_compiled_module)
