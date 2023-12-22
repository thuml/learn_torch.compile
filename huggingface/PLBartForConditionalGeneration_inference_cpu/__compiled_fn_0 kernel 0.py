
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


cpp_fused_add_embedding_mul_native_layer_norm_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x1 + (768L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 50005);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 50005L), "index out of bounds: 0 <= tmp3 < 50005L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                        auto tmp5 = static_cast<float>(27.712812921102035);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp9);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x1 + (768L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = out_ptr1[static_cast<long>(x0)];
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 50005);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 50005L), "index out of bounds: 0 <= tmp3 < 50005L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                    auto tmp5 = static_cast<float>(27.712812921102035);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 - tmp11;
                    auto tmp14 = static_cast<float>(768.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp16 = static_cast<float>(1e-05);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = 1 / std::sqrt(tmp17);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp12 * tmp19;
                    auto tmp22 = tmp20 * tmp21;
                    auto tmp24 = tmp22 + tmp23;
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_clone_copy_embedding_eq_fill_masked_fill_mul_native_layer_norm_ne_select_scatter_slice_scatter_sum_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    long tmp_acc0 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp3 = static_cast<long>(1);
                        auto tmp4 = tmp2 ? tmp3 : tmp0;
                        auto tmp5 = tmp4 != tmp3;
                        auto tmp6 = c10::convert<long>(tmp5);
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr2[static_cast<long>(0L)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp3 = out_ptr2[static_cast<long>(0L)];
                        auto tmp22 = in_ptr2[static_cast<long>(x0)];
                        auto tmp33 = in_ptr4[static_cast<long>(1536L + x1 + (768L*x0))];
                        auto tmp0 = c10::convert<int>(x0);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<long>(1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp5)(tmp5 + 1024);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 1024L), "index out of bounds: 0 <= tmp8 < 1024L")
                        auto tmp9 = in_ptr2[static_cast<long>(tmp8)];
                        auto tmp10 = static_cast<long>(-100);
                        auto tmp11 = tmp9 == tmp10;
                        auto tmp12 = tmp11 ? tmp4 : tmp9;
                        auto tmp13 = c10::convert<long>(x0);
                        auto tmp14 = tmp13 >= tmp4;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr2[static_cast<long>((-1L) + x0)];
                            auto tmp17 = static_cast<long>(-100);
                            auto tmp18 = tmp16 == tmp17;
                            auto tmp19 = static_cast<long>(1);
                            auto tmp20 = tmp18 ? tmp19 : tmp16;
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0);
                        auto tmp23 = tmp22 == tmp10;
                        auto tmp24 = tmp23 ? tmp4 : tmp22;
                        auto tmp25 = tmp14 ? tmp21 : tmp24;
                        auto tmp26 = tmp2 ? tmp12 : tmp25;
                        auto tmp27 = decltype(tmp26)(tmp26 + 50005);
                        auto tmp28 = tmp26 < 0;
                        auto tmp29 = tmp28 ? tmp27 : tmp26;
                        TORCH_CHECK((0 <= tmp29) & (tmp29 < 50005L), "index out of bounds: 0 <= tmp29 < 50005L")
                        auto tmp30 = in_ptr3[static_cast<long>(x1 + (768L*tmp29))];
                        auto tmp31 = static_cast<float>(27.712812921102035);
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        auto tmp34 = decltype(tmp32)(tmp32 + tmp33);
                        out_ptr3[static_cast<long>(x1 + (768L*x0))] = tmp34;
                        tmp_acc0 = welford_combine(tmp_acc0, tmp34);
                    }
                    out_ptr4[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr5[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr4[static_cast<long>(x0)];
                    auto tmp4 = out_ptr5[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr6 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
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
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr2[static_cast<long>(x0)];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_86 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_89 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_95 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_97 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_add_nll_loss_forward_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50000L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (50005L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (50005L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(50000L); x1<static_cast<long>(50005L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (50005L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x1)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        in_out_ptr0[static_cast<long>(x1 + (50005L*x0))] = tmp2;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50000L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (50005L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(50000L); x1<static_cast<long>(50005L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (50005L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x0)];
                        auto tmp9 = out_ptr0[static_cast<long>(x0)];
                        auto tmp11 = out_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = static_cast<long>(0);
                        auto tmp4 = tmp2 ? tmp0 : tmp3;
                        auto tmp5 = decltype(tmp4)(tmp4 + 50005);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 50005L), "index out of bounds: 0 <= tmp7 < 50005L")
                        auto tmp8 = in_out_ptr0[static_cast<long>(tmp7 + (50005L*x0))];
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
                in_out_ptr1[static_cast<long>(0L)] = tmp3;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1026, 768), (768, 1))
    assert_size_stride(arg1_1, (1026, 768), (768, 1))
    assert_size_stride(arg2_1, (50005, 768), (768, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, 768), (768, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, 768), (768, 1))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, 768), (768, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (3072, 768), (768, 1))
    assert_size_stride(arg16_1, (3072, ), (1, ))
    assert_size_stride(arg17_1, (768, 3072), (3072, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, 768), (768, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, 768), (768, 1))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, 768), (768, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (3072, 768), (768, 1))
    assert_size_stride(arg32_1, (3072, ), (1, ))
    assert_size_stride(arg33_1, (768, 3072), (3072, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, 768), (768, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, 768), (768, 1))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, 768), (768, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (3072, 768), (768, 1))
    assert_size_stride(arg48_1, (3072, ), (1, ))
    assert_size_stride(arg49_1, (768, 3072), (3072, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, 768), (768, 1))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, 768), (768, 1))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, 768), (768, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, 768), (768, 1))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (3072, 768), (768, 1))
    assert_size_stride(arg64_1, (3072, ), (1, ))
    assert_size_stride(arg65_1, (768, 3072), (3072, 1))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, 768), (768, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, 768), (768, 1))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, 768), (768, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, 768), (768, 1))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (3072, 768), (768, 1))
    assert_size_stride(arg80_1, (3072, ), (1, ))
    assert_size_stride(arg81_1, (768, 3072), (3072, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, 768), (768, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, 768), (768, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, 768), (768, 1))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (3072, 768), (768, 1))
    assert_size_stride(arg96_1, (3072, ), (1, ))
    assert_size_stride(arg97_1, (768, 3072), (3072, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (50005, 768), (768, 1))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, 768), (768, 1))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, 768), (768, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, 768), (768, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, 768), (768, 1))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, 768), (768, 1))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, 768), (768, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, 768), (768, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, 768), (768, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (3072, 768), (768, 1))
    assert_size_stride(arg125_1, (3072, ), (1, ))
    assert_size_stride(arg126_1, (768, 3072), (3072, 1))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, 768), (768, 1))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, 768), (768, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, 768), (768, 1))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, 768), (768, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, 768), (768, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, 768), (768, 1))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, 768), (768, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, 768), (768, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (3072, 768), (768, 1))
    assert_size_stride(arg151_1, (3072, ), (1, ))
    assert_size_stride(arg152_1, (768, 3072), (3072, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, 768), (768, 1))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, 768), (768, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, 768), (768, 1))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, 768), (768, 1))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, 768), (768, 1))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, 768), (768, 1))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, 768), (768, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, 768), (768, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (3072, 768), (768, 1))
    assert_size_stride(arg177_1, (3072, ), (1, ))
    assert_size_stride(arg178_1, (768, 3072), (3072, 1))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, 768), (768, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, 768), (768, 1))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, 768), (768, 1))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, 768), (768, 1))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (768, 768), (768, 1))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (768, 768), (768, 1))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, 768), (768, 1))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (768, 768), (768, 1))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (3072, 768), (768, 1))
    assert_size_stride(arg203_1, (3072, ), (1, ))
    assert_size_stride(arg204_1, (768, 3072), (3072, 1))
    assert_size_stride(arg205_1, (768, ), (1, ))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (768, 768), (768, 1))
    assert_size_stride(arg209_1, (768, ), (1, ))
    assert_size_stride(arg210_1, (768, 768), (768, 1))
    assert_size_stride(arg211_1, (768, ), (1, ))
    assert_size_stride(arg212_1, (768, 768), (768, 1))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (768, 768), (768, 1))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (768, ), (1, ))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (768, 768), (768, 1))
    assert_size_stride(arg219_1, (768, ), (1, ))
    assert_size_stride(arg220_1, (768, 768), (768, 1))
    assert_size_stride(arg221_1, (768, ), (1, ))
    assert_size_stride(arg222_1, (768, 768), (768, 1))
    assert_size_stride(arg223_1, (768, ), (1, ))
    assert_size_stride(arg224_1, (768, 768), (768, 1))
    assert_size_stride(arg225_1, (768, ), (1, ))
    assert_size_stride(arg226_1, (768, ), (1, ))
    assert_size_stride(arg227_1, (768, ), (1, ))
    assert_size_stride(arg228_1, (3072, 768), (768, 1))
    assert_size_stride(arg229_1, (3072, ), (1, ))
    assert_size_stride(arg230_1, (768, 3072), (3072, 1))
    assert_size_stride(arg231_1, (768, ), (1, ))
    assert_size_stride(arg232_1, (768, ), (1, ))
    assert_size_stride(arg233_1, (768, ), (1, ))
    assert_size_stride(arg234_1, (768, 768), (768, 1))
    assert_size_stride(arg235_1, (768, ), (1, ))
    assert_size_stride(arg236_1, (768, 768), (768, 1))
    assert_size_stride(arg237_1, (768, ), (1, ))
    assert_size_stride(arg238_1, (768, 768), (768, 1))
    assert_size_stride(arg239_1, (768, ), (1, ))
    assert_size_stride(arg240_1, (768, 768), (768, 1))
    assert_size_stride(arg241_1, (768, ), (1, ))
    assert_size_stride(arg242_1, (768, ), (1, ))
    assert_size_stride(arg243_1, (768, ), (1, ))
    assert_size_stride(arg244_1, (768, 768), (768, 1))
    assert_size_stride(arg245_1, (768, ), (1, ))
    assert_size_stride(arg246_1, (768, 768), (768, 1))
    assert_size_stride(arg247_1, (768, ), (1, ))
    assert_size_stride(arg248_1, (768, 768), (768, 1))
    assert_size_stride(arg249_1, (768, ), (1, ))
    assert_size_stride(arg250_1, (768, 768), (768, 1))
    assert_size_stride(arg251_1, (768, ), (1, ))
    assert_size_stride(arg252_1, (768, ), (1, ))
    assert_size_stride(arg253_1, (768, ), (1, ))
    assert_size_stride(arg254_1, (3072, 768), (768, 1))
    assert_size_stride(arg255_1, (3072, ), (1, ))
    assert_size_stride(arg256_1, (768, 3072), (3072, 1))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (768, ), (1, ))
    assert_size_stride(arg259_1, (768, ), (1, ))
    assert_size_stride(arg260_1, (50005, 768), (768, 1))
    assert_size_stride(arg261_1, (1, 50005), (50005, 1))
    assert_size_stride(arg262_1, (1, 1024), (1024, 1))
    assert_size_stride(arg263_1, (1, 1024), (1024, 1))
    buf0 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mul_native_layer_norm_0(c_void_p(arg263_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg0_1
    del arg263_1
    del arg2_1
    del arg3_1
    del arg4_1
    buf4 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg6_1, reinterpret_tensor(buf3, (1024, 768), (768, 1), 0), reinterpret_tensor(arg5_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf4)
    del arg5_1
    del arg6_1
    buf5 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf3, (1024, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf5)
    del arg7_1
    del arg8_1
    buf6 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf3, (1024, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
    del arg10_1
    del arg9_1
    buf7 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    buf8 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    buf9 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_1(c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf10 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf7, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf8, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf9, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf11 = buf10[0]
    del buf10
    buf18 = reinterpret_tensor(buf11, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf11  # reuse
    cpp_fused_clone_2(c_void_p(buf18.data_ptr()))
    buf19 = reinterpret_tensor(buf9, (1024, 768), (768, 1), 0); del buf9  # reuse
    # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf18, (1024, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf19)
    del arg11_1
    del arg12_1
    buf20 = buf1; del buf1  # reuse
    buf21 = buf0; del buf0  # reuse
    buf23 = reinterpret_tensor(buf18, (1, 1024, 768), (786432, 768, 1), 0); del buf18  # reuse
    cpp_fused_add_native_layer_norm_3(c_void_p(buf3.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()))
    del arg13_1
    del arg14_1
    buf24 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg16_1, reinterpret_tensor(buf23, (1024, 768), (768, 1), 0), reinterpret_tensor(arg15_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf24)
    del arg15_1
    del arg16_1
    buf25 = reinterpret_tensor(buf24, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf24  # reuse
    cpp_fused_gelu_4(c_void_p(buf25.data_ptr()))
    buf26 = reinterpret_tensor(buf3, (1024, 768), (768, 1), 0); del buf3  # reuse
    # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf25, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg17_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf26)
    del arg17_1
    del arg18_1
    buf27 = buf21; del buf21  # reuse
    buf28 = buf20; del buf20  # reuse
    buf30 = reinterpret_tensor(buf19, (1, 1024, 768), (786432, 768, 1), 0); del buf19  # reuse
    cpp_fused_add_native_layer_norm_5(c_void_p(buf23.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()))
    del arg19_1
    del arg20_1
    buf31 = buf26; del buf26  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg22_1, reinterpret_tensor(buf30, (1024, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf31)
    del arg21_1
    del arg22_1
    buf32 = reinterpret_tensor(buf23, (1024, 768), (768, 1), 0); del buf23  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg24_1, reinterpret_tensor(buf30, (1024, 768), (768, 1), 0), reinterpret_tensor(arg23_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf32)
    del arg23_1
    del arg24_1
    buf33 = reinterpret_tensor(buf8, (1024, 768), (768, 1), 0); del buf8  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg26_1, reinterpret_tensor(buf30, (1024, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf33)
    del arg25_1
    del arg26_1
    buf34 = buf7; del buf7  # reuse
    buf35 = reinterpret_tensor(buf6, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf6  # reuse
    buf36 = reinterpret_tensor(buf5, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf5  # reuse
    cpp_fused_clone_6(c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf37 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf34, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf35, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf36, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf38 = buf37[0]
    del buf37
    buf45 = reinterpret_tensor(buf38, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf38  # reuse
    cpp_fused_clone_7(c_void_p(buf45.data_ptr()))
    buf46 = reinterpret_tensor(buf36, (1024, 768), (768, 1), 0); del buf36  # reuse
    # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg28_1, reinterpret_tensor(buf45, (1024, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf46)
    del arg27_1
    del arg28_1
    buf47 = buf28; del buf28  # reuse
    buf48 = buf27; del buf27  # reuse
    buf50 = reinterpret_tensor(buf45, (1, 1024, 768), (786432, 768, 1), 0); del buf45  # reuse
    cpp_fused_add_native_layer_norm_8(c_void_p(buf30.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()))
    del arg29_1
    del arg30_1
    buf51 = reinterpret_tensor(buf25, (1024, 3072), (3072, 1), 0); del buf25  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg32_1, reinterpret_tensor(buf50, (1024, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf51)
    del arg31_1
    del arg32_1
    buf52 = reinterpret_tensor(buf51, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf51  # reuse
    cpp_fused_gelu_9(c_void_p(buf52.data_ptr()))
    buf53 = buf46; del buf46  # reuse
    # Source Nodes: [hidden_states_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg34_1, reinterpret_tensor(buf52, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg33_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf53)
    del arg33_1
    del arg34_1
    buf54 = buf48; del buf48  # reuse
    buf55 = buf47; del buf47  # reuse
    buf57 = buf30; del buf30  # reuse
    cpp_fused_add_native_layer_norm_10(c_void_p(buf50.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg35_1
    del arg36_1
    buf58 = buf53; del buf53  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_2_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg38_1, reinterpret_tensor(buf57, (1024, 768), (768, 1), 0), reinterpret_tensor(arg37_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf58)
    del arg37_1
    del arg38_1
    buf59 = reinterpret_tensor(buf50, (1024, 768), (768, 1), 0); del buf50  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_2_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg40_1, reinterpret_tensor(buf57, (1024, 768), (768, 1), 0), reinterpret_tensor(arg39_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf59)
    del arg39_1
    del arg40_1
    buf60 = reinterpret_tensor(buf35, (1024, 768), (768, 1), 0); del buf35  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_2_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg42_1, reinterpret_tensor(buf57, (1024, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf60)
    del arg41_1
    del arg42_1
    buf61 = buf34; del buf34  # reuse
    buf62 = reinterpret_tensor(buf33, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf33  # reuse
    buf63 = reinterpret_tensor(buf32, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf32  # reuse
    cpp_fused_clone_11(c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf64 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf61, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf62, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf63, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf65 = buf64[0]
    del buf64
    buf72 = reinterpret_tensor(buf65, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf65  # reuse
    cpp_fused_clone_12(c_void_p(buf72.data_ptr()))
    buf73 = reinterpret_tensor(buf63, (1024, 768), (768, 1), 0); del buf63  # reuse
    # Source Nodes: [hidden_states_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg44_1, reinterpret_tensor(buf72, (1024, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf73)
    del arg43_1
    del arg44_1
    buf74 = buf55; del buf55  # reuse
    buf75 = buf54; del buf54  # reuse
    buf77 = reinterpret_tensor(buf72, (1, 1024, 768), (786432, 768, 1), 0); del buf72  # reuse
    cpp_fused_add_native_layer_norm_13(c_void_p(buf57.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()))
    del arg45_1
    del arg46_1
    buf78 = reinterpret_tensor(buf52, (1024, 3072), (3072, 1), 0); del buf52  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_2_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg48_1, reinterpret_tensor(buf77, (1024, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf78)
    del arg47_1
    del arg48_1
    buf79 = reinterpret_tensor(buf78, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf78  # reuse
    cpp_fused_gelu_14(c_void_p(buf79.data_ptr()))
    buf80 = buf73; del buf73  # reuse
    # Source Nodes: [hidden_states_31], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg50_1, reinterpret_tensor(buf79, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg49_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf80)
    del arg49_1
    del arg50_1
    buf81 = buf75; del buf75  # reuse
    buf82 = buf74; del buf74  # reuse
    buf84 = buf57; del buf57  # reuse
    cpp_fused_add_native_layer_norm_15(c_void_p(buf77.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()))
    del arg51_1
    del arg52_1
    buf85 = buf80; del buf80  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg54_1, reinterpret_tensor(buf84, (1024, 768), (768, 1), 0), reinterpret_tensor(arg53_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf85)
    del arg53_1
    del arg54_1
    buf86 = reinterpret_tensor(buf77, (1024, 768), (768, 1), 0); del buf77  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg56_1, reinterpret_tensor(buf84, (1024, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf86)
    del arg55_1
    del arg56_1
    buf87 = reinterpret_tensor(buf62, (1024, 768), (768, 1), 0); del buf62  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg58_1, reinterpret_tensor(buf84, (1024, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf87)
    del arg57_1
    del arg58_1
    buf88 = buf61; del buf61  # reuse
    buf89 = reinterpret_tensor(buf60, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf60  # reuse
    buf90 = reinterpret_tensor(buf59, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf59  # reuse
    cpp_fused_clone_16(c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf91 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf88, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf89, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf90, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf92 = buf91[0]
    del buf91
    buf99 = reinterpret_tensor(buf92, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf92  # reuse
    cpp_fused_clone_17(c_void_p(buf99.data_ptr()))
    buf100 = reinterpret_tensor(buf90, (1024, 768), (768, 1), 0); del buf90  # reuse
    # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg60_1, reinterpret_tensor(buf99, (1024, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf100)
    del arg59_1
    del arg60_1
    buf101 = buf82; del buf82  # reuse
    buf102 = buf81; del buf81  # reuse
    buf104 = reinterpret_tensor(buf99, (1, 1024, 768), (786432, 768, 1), 0); del buf99  # reuse
    cpp_fused_add_native_layer_norm_18(c_void_p(buf84.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()))
    del arg61_1
    del arg62_1
    buf105 = reinterpret_tensor(buf79, (1024, 3072), (3072, 1), 0); del buf79  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg64_1, reinterpret_tensor(buf104, (1024, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf105)
    del arg63_1
    del arg64_1
    buf106 = reinterpret_tensor(buf105, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf105  # reuse
    cpp_fused_gelu_19(c_void_p(buf106.data_ptr()))
    buf107 = reinterpret_tensor(buf84, (1024, 768), (768, 1), 0); del buf84  # reuse
    # Source Nodes: [hidden_states_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg66_1, reinterpret_tensor(buf106, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg65_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf107)
    del arg65_1
    del arg66_1
    buf108 = buf102; del buf102  # reuse
    buf109 = buf101; del buf101  # reuse
    buf111 = reinterpret_tensor(buf100, (1, 1024, 768), (786432, 768, 1), 0); del buf100  # reuse
    cpp_fused_add_native_layer_norm_20(c_void_p(buf104.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()))
    del arg67_1
    del arg68_1
    buf112 = buf107; del buf107  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_4_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg70_1, reinterpret_tensor(buf111, (1024, 768), (768, 1), 0), reinterpret_tensor(arg69_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf112)
    del arg69_1
    del arg70_1
    buf113 = reinterpret_tensor(buf104, (1024, 768), (768, 1), 0); del buf104  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_4_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg72_1, reinterpret_tensor(buf111, (1024, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf113)
    del arg71_1
    del arg72_1
    buf114 = reinterpret_tensor(buf89, (1024, 768), (768, 1), 0); del buf89  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_4_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg74_1, reinterpret_tensor(buf111, (1024, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf114)
    del arg73_1
    del arg74_1
    buf115 = buf88; del buf88  # reuse
    buf116 = reinterpret_tensor(buf87, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf87  # reuse
    buf117 = reinterpret_tensor(buf86, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf86  # reuse
    cpp_fused_clone_21(c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf118 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf115, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf116, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf117, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf119 = buf118[0]
    del buf118
    buf126 = reinterpret_tensor(buf119, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf119  # reuse
    cpp_fused_clone_22(c_void_p(buf126.data_ptr()))
    buf127 = reinterpret_tensor(buf117, (1024, 768), (768, 1), 0); del buf117  # reuse
    # Source Nodes: [hidden_states_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg76_1, reinterpret_tensor(buf126, (1024, 768), (768, 1), 0), reinterpret_tensor(arg75_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf127)
    del arg75_1
    del arg76_1
    buf128 = buf109; del buf109  # reuse
    buf129 = buf108; del buf108  # reuse
    buf131 = reinterpret_tensor(buf126, (1, 1024, 768), (786432, 768, 1), 0); del buf126  # reuse
    cpp_fused_add_native_layer_norm_23(c_void_p(buf111.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf131.data_ptr()))
    del arg77_1
    del arg78_1
    buf132 = reinterpret_tensor(buf106, (1024, 3072), (3072, 1), 0); del buf106  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_4_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg80_1, reinterpret_tensor(buf131, (1024, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf132)
    del arg79_1
    del arg80_1
    buf133 = reinterpret_tensor(buf132, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf132  # reuse
    cpp_fused_gelu_24(c_void_p(buf133.data_ptr()))
    buf134 = buf127; del buf127  # reuse
    # Source Nodes: [hidden_states_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg82_1, reinterpret_tensor(buf133, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg81_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf134)
    del arg81_1
    del arg82_1
    buf135 = buf129; del buf129  # reuse
    buf136 = buf128; del buf128  # reuse
    buf138 = buf111; del buf111  # reuse
    cpp_fused_add_native_layer_norm_25(c_void_p(buf131.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()))
    del arg83_1
    del arg84_1
    buf139 = buf134; del buf134  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg86_1, reinterpret_tensor(buf138, (1024, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf139)
    del arg85_1
    del arg86_1
    buf140 = reinterpret_tensor(buf131, (1024, 768), (768, 1), 0); del buf131  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg88_1, reinterpret_tensor(buf138, (1024, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf140)
    del arg87_1
    del arg88_1
    buf141 = reinterpret_tensor(buf116, (1024, 768), (768, 1), 0); del buf116  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg90_1, reinterpret_tensor(buf138, (1024, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf141)
    del arg89_1
    del arg90_1
    buf142 = buf115; del buf115  # reuse
    buf143 = reinterpret_tensor(buf114, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf114  # reuse
    buf144 = reinterpret_tensor(buf113, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf113  # reuse
    cpp_fused_clone_26(c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf145 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf142, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf143, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf144, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf146 = buf145[0]
    del buf145
    buf153 = reinterpret_tensor(buf146, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf146  # reuse
    cpp_fused_clone_27(c_void_p(buf153.data_ptr()))
    buf154 = reinterpret_tensor(buf144, (1024, 768), (768, 1), 0); del buf144  # reuse
    # Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg92_1, reinterpret_tensor(buf153, (1024, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf154)
    del arg91_1
    del arg92_1
    buf155 = buf136; del buf136  # reuse
    buf156 = buf135; del buf135  # reuse
    buf158 = reinterpret_tensor(buf153, (1, 1024, 768), (786432, 768, 1), 0); del buf153  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf138.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf158.data_ptr()))
    del arg93_1
    del arg94_1
    buf159 = reinterpret_tensor(buf133, (1024, 3072), (3072, 1), 0); del buf133  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg96_1, reinterpret_tensor(buf158, (1024, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf159)
    del arg95_1
    del arg96_1
    buf160 = reinterpret_tensor(buf159, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf159  # reuse
    cpp_fused_gelu_29(c_void_p(buf160.data_ptr()))
    buf161 = buf154; del buf154  # reuse
    # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg98_1, reinterpret_tensor(buf160, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg97_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf161)
    del arg97_1
    del arg98_1
    buf162 = buf156; del buf156  # reuse
    buf163 = buf155; del buf155  # reuse
    buf165 = empty((1, ), device='cpu', dtype=torch.int64)
    buf166 = buf138; del buf138  # reuse
    buf167 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf168 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf170 = reinterpret_tensor(buf143, (1, 1024, 768), (786432, 768, 1), 0); del buf143  # reuse
    cpp_fused_add_clone_copy_embedding_eq_fill_masked_fill_mul_native_layer_norm_ne_select_scatter_slice_scatter_sum_30(c_void_p(buf158.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()))
    del arg101_1
    del arg102_1
    del arg103_1
    del arg1_1
    buf171 = reinterpret_tensor(buf166, (1024, 768), (768, 1), 0); del buf166  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg105_1, reinterpret_tensor(buf170, (1024, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf171)
    del arg104_1
    del arg105_1
    buf172 = reinterpret_tensor(buf142, (1024, 768), (768, 1), 0); del buf142  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg107_1, reinterpret_tensor(buf170, (1024, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf172)
    del arg106_1
    del arg107_1
    buf173 = reinterpret_tensor(buf141, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf141  # reuse
    buf174 = reinterpret_tensor(buf140, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf140  # reuse
    cpp_fused_clone_31(c_void_p(buf172.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    buf175 = empty((12, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf174, (12, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf173, (12, 64, 1024), (65536, 1, 64), 0), out=buf175)
    buf176 = empty_strided((12, 1024, 1), (1024, 1, 12288), device='cpu', dtype=torch.float32)
    buf177 = buf175; del buf175  # reuse
    buf178 = empty_strided((12, 1024, 1), (1024, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_32(c_void_p(buf177.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf178.data_ptr()))
    buf179 = reinterpret_tensor(buf174, (1024, 768), (768, 1), 0); del buf174  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg109_1, reinterpret_tensor(buf170, (1024, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf179)
    del arg108_1
    del arg109_1
    buf180 = reinterpret_tensor(buf172, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf172  # reuse
    buf181 = buf177; del buf177  # reuse
    cpp_fused__softmax_clone_33(c_void_p(buf181.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()))
    buf182 = reinterpret_tensor(buf179, (12, 1024, 64), (65536, 64, 1), 0); del buf179  # reuse
    # Source Nodes: [attn_output_30, attn_weights_15], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf181, reinterpret_tensor(buf180, (12, 1024, 64), (65536, 64, 1), 0), out=buf182)
    buf183 = reinterpret_tensor(buf171, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf171  # reuse
    cpp_fused_clone_34(c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    buf184 = reinterpret_tensor(buf182, (1024, 768), (768, 1), 0); del buf182  # reuse
    # Source Nodes: [hidden_states_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg111_1, reinterpret_tensor(buf183, (1024, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf184)
    del arg110_1
    del arg111_1
    buf185 = buf168; del buf168  # reuse
    buf186 = buf167; del buf167  # reuse
    buf188 = reinterpret_tensor(buf183, (1, 1024, 768), (786432, 768, 1), 0); del buf183  # reuse
    cpp_fused_add_native_layer_norm_35(c_void_p(buf170.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf188.data_ptr()))
    del arg112_1
    del arg113_1
    del buf185
    del buf186
    buf189 = buf184; del buf184  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg115_1, reinterpret_tensor(buf188, (1024, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf189)
    del arg114_1
    del arg115_1
    buf190 = buf170; del buf170  # reuse
    cpp_fused_add_native_layer_norm_36(c_void_p(buf158.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(buf190.data_ptr()))
    del arg100_1
    del arg99_1
    buf191 = buf161; del buf161  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg117_1, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf191)
    del arg116_1
    del arg117_1
    buf192 = reinterpret_tensor(buf158, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf158  # reuse
    cpp_fused_clone_37(c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()))
    buf193 = buf191; del buf191  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg119_1, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf193)
    del arg118_1
    del arg119_1
    buf194 = reinterpret_tensor(buf139, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf139  # reuse
    buf195 = reinterpret_tensor(buf112, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf112  # reuse
    cpp_fused_clone_38(c_void_p(buf193.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf196 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf195, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf192, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), reinterpret_tensor(buf194, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), scale=1.0)
    buf197 = buf196[0]
    del buf196
    buf204 = reinterpret_tensor(buf197, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf197  # reuse
    cpp_fused_clone_39(c_void_p(buf204.data_ptr()))
    buf205 = reinterpret_tensor(buf195, (1024, 768), (768, 1), 0); del buf195  # reuse
    # Source Nodes: [hidden_states_76], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg121_1, reinterpret_tensor(buf204, (1024, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf205)
    del arg120_1
    del arg121_1
    buf206 = buf163; del buf163  # reuse
    buf207 = buf162; del buf162  # reuse
    buf209 = reinterpret_tensor(buf204, (1, 1024, 768), (786432, 768, 1), 0); del buf204  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf188.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf209.data_ptr()))
    del arg122_1
    del arg123_1
    buf210 = reinterpret_tensor(buf160, (1024, 3072), (3072, 1), 0); del buf160  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg125_1, reinterpret_tensor(buf209, (1024, 768), (768, 1), 0), reinterpret_tensor(arg124_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf210)
    del arg124_1
    del arg125_1
    buf211 = reinterpret_tensor(buf210, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf210  # reuse
    cpp_fused_gelu_41(c_void_p(buf211.data_ptr()))
    buf212 = buf205; del buf205  # reuse
    # Source Nodes: [hidden_states_82], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg127_1, reinterpret_tensor(buf211, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg126_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf212)
    del arg126_1
    del arg127_1
    buf213 = buf207; del buf207  # reuse
    buf214 = buf206; del buf206  # reuse
    buf216 = buf188; del buf188  # reuse
    cpp_fused_add_native_layer_norm_42(c_void_p(buf209.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()))
    del arg128_1
    del arg129_1
    buf217 = buf212; del buf212  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg131_1, reinterpret_tensor(buf216, (1024, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf217)
    del arg130_1
    del arg131_1
    buf218 = reinterpret_tensor(buf209, (1024, 768), (768, 1), 0); del buf209  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg133_1, reinterpret_tensor(buf216, (1024, 768), (768, 1), 0), reinterpret_tensor(arg132_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf218)
    del arg132_1
    del arg133_1
    buf219 = reinterpret_tensor(buf193, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf193  # reuse
    buf220 = reinterpret_tensor(buf189, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf189  # reuse
    cpp_fused_clone_43(c_void_p(buf218.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()))
    buf221 = buf181; del buf181  # reuse
    # Source Nodes: [attn_weights_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf220, (12, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf219, (12, 64, 1024), (65536, 1, 64), 0), out=buf221)
    buf222 = buf178; del buf178  # reuse
    buf223 = buf221; del buf221  # reuse
    buf224 = buf176; del buf176  # reuse
    cpp_fused__softmax_44(c_void_p(buf223.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf224.data_ptr()))
    buf225 = reinterpret_tensor(buf220, (1024, 768), (768, 1), 0); del buf220  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg135_1, reinterpret_tensor(buf216, (1024, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf225)
    del arg134_1
    del arg135_1
    buf226 = reinterpret_tensor(buf218, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf218  # reuse
    buf227 = buf223; del buf223  # reuse
    cpp_fused__softmax_clone_45(c_void_p(buf227.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()))
    buf228 = reinterpret_tensor(buf225, (12, 1024, 64), (65536, 64, 1), 0); del buf225  # reuse
    # Source Nodes: [attn_output_40, attn_weights_21], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf227, reinterpret_tensor(buf226, (12, 1024, 64), (65536, 64, 1), 0), out=buf228)
    buf229 = reinterpret_tensor(buf217, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf217  # reuse
    cpp_fused_clone_46(c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()))
    buf230 = reinterpret_tensor(buf228, (1024, 768), (768, 1), 0); del buf228  # reuse
    # Source Nodes: [hidden_states_87], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg137_1, reinterpret_tensor(buf229, (1024, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf230)
    del arg136_1
    del arg137_1
    buf231 = buf214; del buf214  # reuse
    buf232 = buf213; del buf213  # reuse
    buf234 = reinterpret_tensor(buf229, (1, 1024, 768), (786432, 768, 1), 0); del buf229  # reuse
    cpp_fused_add_native_layer_norm_47(c_void_p(buf216.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()))
    del arg138_1
    del arg139_1
    buf235 = buf230; del buf230  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg141_1, reinterpret_tensor(buf234, (1024, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf235)
    del arg140_1
    del arg141_1
    buf236 = reinterpret_tensor(buf216, (1024, 768), (768, 1), 0); del buf216  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf236)
    del arg142_1
    del arg143_1
    buf237 = reinterpret_tensor(buf85, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf85  # reuse
    cpp_fused_clone_48(c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()))
    buf238 = buf236; del buf236  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg145_1, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf238)
    del arg144_1
    del arg145_1
    buf239 = reinterpret_tensor(buf58, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf58  # reuse
    buf240 = reinterpret_tensor(buf31, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf31  # reuse
    cpp_fused_clone_49(c_void_p(buf238.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf241 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf240, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf237, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), reinterpret_tensor(buf239, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), scale=1.0)
    buf242 = buf241[0]
    del buf241
    buf249 = reinterpret_tensor(buf242, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf242  # reuse
    cpp_fused_clone_50(c_void_p(buf249.data_ptr()))
    buf250 = reinterpret_tensor(buf240, (1024, 768), (768, 1), 0); del buf240  # reuse
    # Source Nodes: [hidden_states_91], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg147_1, reinterpret_tensor(buf249, (1024, 768), (768, 1), 0), reinterpret_tensor(arg146_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf250)
    del arg146_1
    del arg147_1
    buf251 = buf232; del buf232  # reuse
    buf252 = buf231; del buf231  # reuse
    buf254 = reinterpret_tensor(buf249, (1, 1024, 768), (786432, 768, 1), 0); del buf249  # reuse
    cpp_fused_add_native_layer_norm_51(c_void_p(buf234.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf254.data_ptr()))
    del arg148_1
    del arg149_1
    buf255 = reinterpret_tensor(buf211, (1024, 3072), (3072, 1), 0); del buf211  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg151_1, reinterpret_tensor(buf254, (1024, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf255)
    del arg150_1
    del arg151_1
    buf256 = reinterpret_tensor(buf255, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf255  # reuse
    cpp_fused_gelu_52(c_void_p(buf256.data_ptr()))
    buf257 = buf250; del buf250  # reuse
    # Source Nodes: [hidden_states_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg153_1, reinterpret_tensor(buf256, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg152_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf257)
    del arg152_1
    del arg153_1
    buf258 = buf252; del buf252  # reuse
    buf259 = buf251; del buf251  # reuse
    buf261 = buf234; del buf234  # reuse
    cpp_fused_add_native_layer_norm_53(c_void_p(buf254.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()))
    del arg154_1
    del arg155_1
    buf262 = buf257; del buf257  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg157_1, reinterpret_tensor(buf261, (1024, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf262)
    del arg156_1
    del arg157_1
    buf263 = reinterpret_tensor(buf254, (1024, 768), (768, 1), 0); del buf254  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg159_1, reinterpret_tensor(buf261, (1024, 768), (768, 1), 0), reinterpret_tensor(arg158_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf263)
    del arg158_1
    del arg159_1
    buf264 = reinterpret_tensor(buf238, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf238  # reuse
    buf265 = reinterpret_tensor(buf235, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf235  # reuse
    cpp_fused_clone_54(c_void_p(buf263.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    buf266 = buf227; del buf227  # reuse
    # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf265, (12, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf264, (12, 64, 1024), (65536, 1, 64), 0), out=buf266)
    buf267 = buf224; del buf224  # reuse
    buf268 = buf266; del buf266  # reuse
    buf269 = buf222; del buf222  # reuse
    cpp_fused__softmax_55(c_void_p(buf268.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf269.data_ptr()))
    buf270 = reinterpret_tensor(buf265, (1024, 768), (768, 1), 0); del buf265  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg161_1, reinterpret_tensor(buf261, (1024, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf270)
    del arg160_1
    del arg161_1
    buf271 = reinterpret_tensor(buf263, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf263  # reuse
    buf272 = buf268; del buf268  # reuse
    cpp_fused__softmax_clone_56(c_void_p(buf272.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf271.data_ptr()))
    buf273 = reinterpret_tensor(buf270, (12, 1024, 64), (65536, 64, 1), 0); del buf270  # reuse
    # Source Nodes: [attn_output_50, attn_weights_27], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf272, reinterpret_tensor(buf271, (12, 1024, 64), (65536, 64, 1), 0), out=buf273)
    buf274 = reinterpret_tensor(buf262, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf262  # reuse
    cpp_fused_clone_57(c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    buf275 = reinterpret_tensor(buf273, (1024, 768), (768, 1), 0); del buf273  # reuse
    # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg163_1, reinterpret_tensor(buf274, (1024, 768), (768, 1), 0), reinterpret_tensor(arg162_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf275)
    del arg162_1
    del arg163_1
    buf276 = buf259; del buf259  # reuse
    buf277 = buf258; del buf258  # reuse
    buf279 = reinterpret_tensor(buf274, (1, 1024, 768), (786432, 768, 1), 0); del buf274  # reuse
    cpp_fused_add_native_layer_norm_58(c_void_p(buf261.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf279.data_ptr()))
    del arg164_1
    del arg165_1
    buf280 = buf275; del buf275  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg167_1, reinterpret_tensor(buf279, (1024, 768), (768, 1), 0), reinterpret_tensor(arg166_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf280)
    del arg166_1
    del arg167_1
    buf281 = reinterpret_tensor(buf261, (1024, 768), (768, 1), 0); del buf261  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg169_1, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), reinterpret_tensor(arg168_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf281)
    del arg168_1
    del arg169_1
    buf282 = reinterpret_tensor(buf4, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf4  # reuse
    cpp_fused_clone_59(c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()))
    buf283 = buf281; del buf281  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg171_1, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf283)
    del arg170_1
    del arg171_1
    buf284 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    buf285 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_60(c_void_p(buf283.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf286 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf285, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf282, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), reinterpret_tensor(buf284, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), scale=1.0)
    buf287 = buf286[0]
    del buf286
    buf294 = reinterpret_tensor(buf287, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf287  # reuse
    cpp_fused_clone_61(c_void_p(buf294.data_ptr()))
    buf295 = reinterpret_tensor(buf285, (1024, 768), (768, 1), 0); del buf285  # reuse
    # Source Nodes: [hidden_states_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg173_1, reinterpret_tensor(buf294, (1024, 768), (768, 1), 0), reinterpret_tensor(arg172_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf295)
    del arg172_1
    del arg173_1
    buf296 = buf277; del buf277  # reuse
    buf297 = buf276; del buf276  # reuse
    buf299 = reinterpret_tensor(buf294, (1, 1024, 768), (786432, 768, 1), 0); del buf294  # reuse
    cpp_fused_add_native_layer_norm_62(c_void_p(buf279.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()))
    del arg174_1
    del arg175_1
    buf300 = reinterpret_tensor(buf256, (1024, 3072), (3072, 1), 0); del buf256  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg177_1, reinterpret_tensor(buf299, (1024, 768), (768, 1), 0), reinterpret_tensor(arg176_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf300)
    del arg176_1
    del arg177_1
    buf301 = reinterpret_tensor(buf300, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf300  # reuse
    cpp_fused_gelu_63(c_void_p(buf301.data_ptr()))
    buf302 = buf295; del buf295  # reuse
    # Source Nodes: [hidden_states_112], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg179_1, reinterpret_tensor(buf301, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg178_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf302)
    del arg178_1
    del arg179_1
    buf303 = buf297; del buf297  # reuse
    buf304 = buf296; del buf296  # reuse
    buf306 = buf279; del buf279  # reuse
    cpp_fused_add_native_layer_norm_64(c_void_p(buf299.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()))
    del arg180_1
    del arg181_1
    buf307 = buf302; del buf302  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg183_1, reinterpret_tensor(buf306, (1024, 768), (768, 1), 0), reinterpret_tensor(arg182_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf307)
    del arg182_1
    del arg183_1
    buf308 = reinterpret_tensor(buf299, (1024, 768), (768, 1), 0); del buf299  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg185_1, reinterpret_tensor(buf306, (1024, 768), (768, 1), 0), reinterpret_tensor(arg184_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf308)
    del arg184_1
    del arg185_1
    buf309 = reinterpret_tensor(buf283, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf283  # reuse
    buf310 = reinterpret_tensor(buf280, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf280  # reuse
    cpp_fused_clone_65(c_void_p(buf308.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()))
    buf311 = buf272; del buf272  # reuse
    # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf310, (12, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf309, (12, 64, 1024), (65536, 1, 64), 0), out=buf311)
    buf312 = buf269; del buf269  # reuse
    buf313 = buf311; del buf311  # reuse
    buf314 = buf267; del buf267  # reuse
    cpp_fused__softmax_66(c_void_p(buf313.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf314.data_ptr()))
    buf315 = reinterpret_tensor(buf310, (1024, 768), (768, 1), 0); del buf310  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg187_1, reinterpret_tensor(buf306, (1024, 768), (768, 1), 0), reinterpret_tensor(arg186_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf315)
    del arg186_1
    del arg187_1
    buf316 = reinterpret_tensor(buf308, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf308  # reuse
    buf317 = buf313; del buf313  # reuse
    cpp_fused__softmax_clone_67(c_void_p(buf317.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf316.data_ptr()))
    buf318 = reinterpret_tensor(buf315, (12, 1024, 64), (65536, 64, 1), 0); del buf315  # reuse
    # Source Nodes: [attn_output_60, attn_weights_33], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf317, reinterpret_tensor(buf316, (12, 1024, 64), (65536, 64, 1), 0), out=buf318)
    buf319 = reinterpret_tensor(buf307, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf307  # reuse
    cpp_fused_clone_68(c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()))
    buf320 = reinterpret_tensor(buf318, (1024, 768), (768, 1), 0); del buf318  # reuse
    # Source Nodes: [hidden_states_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg189_1, reinterpret_tensor(buf319, (1024, 768), (768, 1), 0), reinterpret_tensor(arg188_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf320)
    del arg188_1
    del arg189_1
    buf321 = buf304; del buf304  # reuse
    buf322 = buf303; del buf303  # reuse
    buf324 = reinterpret_tensor(buf319, (1, 1024, 768), (786432, 768, 1), 0); del buf319  # reuse
    cpp_fused_add_native_layer_norm_69(c_void_p(buf306.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()))
    del arg190_1
    del arg191_1
    buf325 = buf320; del buf320  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg193_1, reinterpret_tensor(buf324, (1024, 768), (768, 1), 0), reinterpret_tensor(arg192_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf325)
    del arg192_1
    del arg193_1
    buf326 = reinterpret_tensor(buf306, (1024, 768), (768, 1), 0); del buf306  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg195_1, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), reinterpret_tensor(arg194_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf326)
    del arg194_1
    del arg195_1
    buf327 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_70(c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    buf328 = buf326; del buf326  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg197_1, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), reinterpret_tensor(arg196_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf328)
    del arg196_1
    del arg197_1
    buf329 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    buf330 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_71(c_void_p(buf328.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf331 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf330, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf327, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), reinterpret_tensor(buf329, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), scale=1.0)
    buf332 = buf331[0]
    del buf331
    buf339 = reinterpret_tensor(buf332, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf332  # reuse
    cpp_fused_clone_72(c_void_p(buf339.data_ptr()))
    buf340 = reinterpret_tensor(buf330, (1024, 768), (768, 1), 0); del buf330  # reuse
    # Source Nodes: [hidden_states_121], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg199_1, reinterpret_tensor(buf339, (1024, 768), (768, 1), 0), reinterpret_tensor(arg198_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf340)
    del arg198_1
    del arg199_1
    buf341 = buf322; del buf322  # reuse
    buf342 = buf321; del buf321  # reuse
    buf344 = reinterpret_tensor(buf339, (1, 1024, 768), (786432, 768, 1), 0); del buf339  # reuse
    cpp_fused_add_native_layer_norm_73(c_void_p(buf324.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf344.data_ptr()))
    del arg200_1
    del arg201_1
    buf345 = reinterpret_tensor(buf301, (1024, 3072), (3072, 1), 0); del buf301  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg203_1, reinterpret_tensor(buf344, (1024, 768), (768, 1), 0), reinterpret_tensor(arg202_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf345)
    del arg202_1
    del arg203_1
    buf346 = reinterpret_tensor(buf345, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf345  # reuse
    cpp_fused_gelu_74(c_void_p(buf346.data_ptr()))
    buf347 = buf340; del buf340  # reuse
    # Source Nodes: [hidden_states_127], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg205_1, reinterpret_tensor(buf346, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg204_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf347)
    del arg204_1
    del arg205_1
    buf348 = buf342; del buf342  # reuse
    buf349 = buf341; del buf341  # reuse
    buf351 = buf324; del buf324  # reuse
    cpp_fused_add_native_layer_norm_75(c_void_p(buf344.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf351.data_ptr()))
    del arg206_1
    del arg207_1
    buf352 = buf347; del buf347  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg209_1, reinterpret_tensor(buf351, (1024, 768), (768, 1), 0), reinterpret_tensor(arg208_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf352)
    del arg208_1
    del arg209_1
    buf353 = reinterpret_tensor(buf344, (1024, 768), (768, 1), 0); del buf344  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg211_1, reinterpret_tensor(buf351, (1024, 768), (768, 1), 0), reinterpret_tensor(arg210_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf353)
    del arg210_1
    del arg211_1
    buf354 = reinterpret_tensor(buf328, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf328  # reuse
    buf355 = reinterpret_tensor(buf325, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf325  # reuse
    cpp_fused_clone_76(c_void_p(buf353.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()))
    buf356 = buf317; del buf317  # reuse
    # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf355, (12, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf354, (12, 64, 1024), (65536, 1, 64), 0), out=buf356)
    buf357 = buf314; del buf314  # reuse
    buf358 = buf356; del buf356  # reuse
    buf359 = buf312; del buf312  # reuse
    cpp_fused__softmax_77(c_void_p(buf358.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf359.data_ptr()))
    buf360 = reinterpret_tensor(buf355, (1024, 768), (768, 1), 0); del buf355  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg213_1, reinterpret_tensor(buf351, (1024, 768), (768, 1), 0), reinterpret_tensor(arg212_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf360)
    del arg212_1
    del arg213_1
    buf361 = reinterpret_tensor(buf353, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf353  # reuse
    buf362 = buf358; del buf358  # reuse
    cpp_fused__softmax_clone_78(c_void_p(buf362.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf361.data_ptr()))
    buf363 = reinterpret_tensor(buf360, (12, 1024, 64), (65536, 64, 1), 0); del buf360  # reuse
    # Source Nodes: [attn_output_70, attn_weights_39], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf362, reinterpret_tensor(buf361, (12, 1024, 64), (65536, 64, 1), 0), out=buf363)
    buf364 = reinterpret_tensor(buf352, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf352  # reuse
    cpp_fused_clone_79(c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()))
    buf365 = reinterpret_tensor(buf363, (1024, 768), (768, 1), 0); del buf363  # reuse
    # Source Nodes: [hidden_states_132], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg215_1, reinterpret_tensor(buf364, (1024, 768), (768, 1), 0), reinterpret_tensor(arg214_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf365)
    del arg214_1
    del arg215_1
    buf366 = buf349; del buf349  # reuse
    buf367 = buf348; del buf348  # reuse
    buf369 = reinterpret_tensor(buf364, (1, 1024, 768), (786432, 768, 1), 0); del buf364  # reuse
    cpp_fused_add_native_layer_norm_80(c_void_p(buf351.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf369.data_ptr()))
    del arg216_1
    del arg217_1
    buf370 = buf365; del buf365  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg219_1, reinterpret_tensor(buf369, (1024, 768), (768, 1), 0), reinterpret_tensor(arg218_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf370)
    del arg218_1
    del arg219_1
    buf371 = reinterpret_tensor(buf351, (1024, 768), (768, 1), 0); del buf351  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg221_1, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), reinterpret_tensor(arg220_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf371)
    del arg220_1
    del arg221_1
    buf372 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_81(c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()))
    buf373 = buf371; del buf371  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg223_1, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), reinterpret_tensor(arg222_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf373)
    del arg222_1
    del arg223_1
    buf374 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    buf375 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_82(c_void_p(buf373.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf376 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf375, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf372, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), reinterpret_tensor(buf374, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), scale=1.0)
    buf377 = buf376[0]
    del buf376
    buf384 = reinterpret_tensor(buf377, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf377  # reuse
    cpp_fused_clone_83(c_void_p(buf384.data_ptr()))
    buf385 = reinterpret_tensor(buf375, (1024, 768), (768, 1), 0); del buf375  # reuse
    # Source Nodes: [hidden_states_136], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg225_1, reinterpret_tensor(buf384, (1024, 768), (768, 1), 0), reinterpret_tensor(arg224_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf385)
    del arg224_1
    del arg225_1
    buf386 = buf367; del buf367  # reuse
    buf387 = buf366; del buf366  # reuse
    buf389 = reinterpret_tensor(buf384, (1, 1024, 768), (786432, 768, 1), 0); del buf384  # reuse
    cpp_fused_add_native_layer_norm_84(c_void_p(buf369.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf389.data_ptr()))
    del arg226_1
    del arg227_1
    buf390 = reinterpret_tensor(buf346, (1024, 3072), (3072, 1), 0); del buf346  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg229_1, reinterpret_tensor(buf389, (1024, 768), (768, 1), 0), reinterpret_tensor(arg228_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf390)
    del arg228_1
    del arg229_1
    buf391 = reinterpret_tensor(buf390, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf390  # reuse
    cpp_fused_gelu_85(c_void_p(buf391.data_ptr()))
    buf392 = buf385; del buf385  # reuse
    # Source Nodes: [hidden_states_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg231_1, reinterpret_tensor(buf391, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg230_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf392)
    del arg230_1
    del arg231_1
    buf393 = buf387; del buf387  # reuse
    buf394 = buf386; del buf386  # reuse
    buf396 = buf369; del buf369  # reuse
    cpp_fused_add_native_layer_norm_86(c_void_p(buf389.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf396.data_ptr()))
    del arg232_1
    del arg233_1
    buf397 = buf392; del buf392  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg235_1, reinterpret_tensor(buf396, (1024, 768), (768, 1), 0), reinterpret_tensor(arg234_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf397)
    del arg234_1
    del arg235_1
    buf398 = reinterpret_tensor(buf389, (1024, 768), (768, 1), 0); del buf389  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg237_1, reinterpret_tensor(buf396, (1024, 768), (768, 1), 0), reinterpret_tensor(arg236_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf398)
    del arg236_1
    del arg237_1
    buf399 = reinterpret_tensor(buf373, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf373  # reuse
    buf400 = reinterpret_tensor(buf370, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf370  # reuse
    cpp_fused_clone_87(c_void_p(buf398.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()))
    buf401 = buf362; del buf362  # reuse
    # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf400, (12, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf399, (12, 64, 1024), (65536, 1, 64), 0), out=buf401)
    buf402 = buf359; del buf359  # reuse
    buf403 = buf401; del buf401  # reuse
    buf404 = buf357; del buf357  # reuse
    cpp_fused__softmax_88(c_void_p(buf403.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf404.data_ptr()))
    del buf402
    buf405 = reinterpret_tensor(buf400, (1024, 768), (768, 1), 0); del buf400  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg239_1, reinterpret_tensor(buf396, (1024, 768), (768, 1), 0), reinterpret_tensor(arg238_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf405)
    del arg238_1
    del arg239_1
    buf406 = reinterpret_tensor(buf398, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf398  # reuse
    buf407 = buf403; del buf403  # reuse
    cpp_fused__softmax_clone_89(c_void_p(buf407.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf406.data_ptr()))
    del buf404
    buf408 = reinterpret_tensor(buf405, (12, 1024, 64), (65536, 64, 1), 0); del buf405  # reuse
    # Source Nodes: [attn_output_80, attn_weights_45], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf407, reinterpret_tensor(buf406, (12, 1024, 64), (65536, 64, 1), 0), out=buf408)
    del buf407
    buf409 = reinterpret_tensor(buf397, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf397  # reuse
    cpp_fused_clone_90(c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    buf410 = reinterpret_tensor(buf408, (1024, 768), (768, 1), 0); del buf408  # reuse
    # Source Nodes: [hidden_states_147], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg241_1, reinterpret_tensor(buf409, (1024, 768), (768, 1), 0), reinterpret_tensor(arg240_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf410)
    del arg240_1
    del arg241_1
    buf411 = buf394; del buf394  # reuse
    buf412 = buf393; del buf393  # reuse
    buf414 = reinterpret_tensor(buf409, (1, 1024, 768), (786432, 768, 1), 0); del buf409  # reuse
    cpp_fused_add_native_layer_norm_91(c_void_p(buf396.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf414.data_ptr()))
    del arg242_1
    del arg243_1
    buf415 = buf410; del buf410  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg245_1, reinterpret_tensor(buf414, (1024, 768), (768, 1), 0), reinterpret_tensor(arg244_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf415)
    del arg244_1
    del arg245_1
    buf416 = reinterpret_tensor(buf396, (1024, 768), (768, 1), 0); del buf396  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg247_1, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), reinterpret_tensor(arg246_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf416)
    del arg246_1
    del arg247_1
    buf417 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_92(c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()))
    buf418 = buf416; del buf416  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg249_1, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), reinterpret_tensor(arg248_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf418)
    del arg248_1
    del arg249_1
    buf419 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    buf420 = empty((1, 12, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_93(c_void_p(buf418.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()))
    del buf415
    del buf418
    # Source Nodes: [], Original ATen: []
    buf421 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf420, (1, 12, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf417, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), reinterpret_tensor(buf419, (1, 12, 1024, 64), (786432, 65536, 64, 1), 0), scale=1.0)
    buf422 = buf421[0]
    del buf421
    buf429 = reinterpret_tensor(buf422, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf422  # reuse
    cpp_fused_clone_94(c_void_p(buf429.data_ptr()))
    buf430 = reinterpret_tensor(buf420, (1024, 768), (768, 1), 0); del buf420  # reuse
    # Source Nodes: [hidden_states_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg251_1, reinterpret_tensor(buf429, (1024, 768), (768, 1), 0), reinterpret_tensor(arg250_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf430)
    del arg250_1
    del arg251_1
    buf431 = buf412; del buf412  # reuse
    buf432 = buf411; del buf411  # reuse
    buf434 = reinterpret_tensor(buf429, (1, 1024, 768), (786432, 768, 1), 0); del buf429  # reuse
    cpp_fused_add_native_layer_norm_95(c_void_p(buf414.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf434.data_ptr()))
    del arg252_1
    del arg253_1
    buf435 = reinterpret_tensor(buf391, (1024, 3072), (3072, 1), 0); del buf391  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg255_1, reinterpret_tensor(buf434, (1024, 768), (768, 1), 0), reinterpret_tensor(arg254_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf435)
    del arg254_1
    del arg255_1
    buf436 = reinterpret_tensor(buf435, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf435  # reuse
    cpp_fused_gelu_96(c_void_p(buf436.data_ptr()))
    buf437 = buf430; del buf430  # reuse
    # Source Nodes: [hidden_states_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg257_1, reinterpret_tensor(buf436, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg256_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf437)
    del arg256_1
    del arg257_1
    del buf436
    buf438 = buf432; del buf432  # reuse
    buf439 = buf431; del buf431  # reuse
    buf441 = buf414; del buf414  # reuse
    cpp_fused_add_native_layer_norm_97(c_void_p(buf434.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf441.data_ptr()))
    del arg258_1
    del arg259_1
    del buf434
    del buf437
    buf442 = empty((1024, 50005), device='cpu', dtype=torch.float32)
    # Source Nodes: [lm_logits], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf441, (1024, 768), (768, 1), 0), reinterpret_tensor(arg260_1, (768, 50005), (1, 768), 0), out=buf442)
    del arg260_1
    del buf441
    buf443 = reinterpret_tensor(buf442, (1, 1024, 50005), (51205120, 50005, 1), 0); del buf442  # reuse
    buf444 = reinterpret_tensor(buf439, (1024, 1), (1, 1024), 0); del buf439  # reuse
    buf445 = reinterpret_tensor(buf438, (1024, 1), (1, 1024), 0); del buf438  # reuse
    buf446 = empty((), device='cpu', dtype=torch.float32)
    buf447 = reinterpret_tensor(buf165, (), (), 0); del buf165  # reuse
    buf448 = buf446; del buf446  # reuse
    cpp_fused__log_softmax_add_nll_loss_forward_98(c_void_p(buf443.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf447.data_ptr()))
    del arg261_1
    del arg262_1
    return (buf448, buf443, buf173, buf180, buf192, buf194, buf219, buf226, buf237, buf239, buf264, buf271, buf282, buf284, buf309, buf316, buf327, buf329, buf354, buf361, buf372, buf374, buf399, buf406, buf417, buf419, buf190, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1026, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1026, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((50005, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((50005, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((50005, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((1, 50005), (50005, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    arg263_1 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('PLBartForConditionalGeneration', benchmark_compiled_module)
