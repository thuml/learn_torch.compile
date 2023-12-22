
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
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(2048L + x1 + (1024L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 50265);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 50265L), "index out of bounds: 0 <= tmp3 < 50265L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*tmp3)));
                        auto tmp5 = static_cast<float>(1.0);
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
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(2048L + x1 + (1024L*x0)));
                        auto tmp10 = out_ptr0[static_cast<long>(x0)];
                        auto tmp13 = out_ptr1[static_cast<long>(x0)];
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp1 = decltype(tmp0)(tmp0 + 50265);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 50265L), "index out of bounds: 0 <= tmp3 < 50265L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*tmp3)));
                        auto tmp5 = static_cast<float>(1.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 - tmp11;
                        auto tmp14 = static_cast<float>(1024.0);
                        auto tmp15 = tmp13 / tmp14;
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = 1 / std::sqrt(tmp17);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp12 * tmp19;
                        auto tmp22 = tmp20 * tmp21;
                        auto tmp24 = tmp22 + tmp23;
                        tmp24.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp24);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr3[static_cast<long>(x0)];
                    auto tmp4 = out_ptr4[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_gelu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_gelu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_gelu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_gelu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_gelu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_gelu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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


cpp_fused_clone_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_gelu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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


cpp_fused_clone_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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


cpp_fused_clone_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_gelu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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


cpp_fused_clone_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_gelu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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


cpp_fused_clone_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_gelu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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


cpp_fused_clone_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_gelu_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_clone_copy_embedding_eq_fill_masked_fill_mul_native_layer_norm_ne_select_scatter_slice_scatter_sum_view_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const long* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                {
                    long tmp_acc0 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp3 = out_ptr2[static_cast<long>(0L)];
                        auto tmp22 = in_ptr4[static_cast<long>(x0)];
                        auto tmp33 = in_ptr6[static_cast<long>(2048L + x1 + (1024L*x0))];
                        auto tmp0 = c10::convert<int>(x0);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<long>(1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp5)(tmp5 + 1024);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 1024L), "index out of bounds: 0 <= tmp8 < 1024L")
                        auto tmp9 = in_ptr4[static_cast<long>(tmp8)];
                        auto tmp10 = static_cast<long>(-100);
                        auto tmp11 = tmp9 == tmp10;
                        auto tmp12 = tmp11 ? tmp4 : tmp9;
                        auto tmp13 = c10::convert<long>(x0);
                        auto tmp14 = tmp13 >= tmp4;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr4[static_cast<long>((-1L) + x0)];
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
                        auto tmp27 = decltype(tmp26)(tmp26 + 50265);
                        auto tmp28 = tmp26 < 0;
                        auto tmp29 = tmp28 ? tmp27 : tmp26;
                        TORCH_CHECK((0 <= tmp29) & (tmp29 < 50265L), "index out of bounds: 0 <= tmp29 < 50265L")
                        auto tmp30 = in_ptr5[static_cast<long>(x1 + (1024L*tmp29))];
                        auto tmp31 = static_cast<float>(1.0);
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        auto tmp34 = decltype(tmp32)(tmp32 + tmp33);
                        out_ptr3[static_cast<long>(x1 + (1024L*x0))] = tmp34;
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
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = out_ptr4[static_cast<long>(x0)];
                        auto tmp4 = out_ptr5[static_cast<long>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(1024.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr6 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp15);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr8[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr7[static_cast<long>(x0)];
                    auto tmp4 = out_ptr8[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr9 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_64 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_native_layer_norm_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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


cpp_fused_gelu_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_clone_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_75 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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


cpp_fused_clone_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_79 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_gelu_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_85 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_clone_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_89 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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


cpp_fused_gelu_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_clone_92 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_94 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_95 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_96 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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


cpp_fused_clone_97 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_99 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_gelu_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_101 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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


cpp_fused_clone_102 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_103 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_104 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_105 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_106 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_clone_107 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_109 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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


cpp_fused_gelu_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_111 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_clone_112 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_113 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_114 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_115 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_116 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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


cpp_fused_clone_117 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_119 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_gelu_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_121 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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


cpp_fused_clone_122 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_123 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_124 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_125 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_126 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_clone_127 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_129 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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


cpp_fused_gelu_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_131 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_clone_132 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_133 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_134 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_135 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_136 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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


cpp_fused_clone_137 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_139 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_gelu_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_141 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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


cpp_fused_clone_142 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_143 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_144 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_145 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_146 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_clone_147 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_148 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_149 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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


cpp_fused_gelu_150 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_151 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_clone_152 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_153 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_154 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_155 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_156 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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


cpp_fused_clone_157 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_159 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_gelu_160 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_161 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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


cpp_fused_clone_162 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_163 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_164 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_165 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_166 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_clone_167 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_168 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_169 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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


cpp_fused_gelu_170 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_171 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
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


cpp_fused_clone_172 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_173 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_174 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_175 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_176 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
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


cpp_fused_clone_177 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_178 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_179 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
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


cpp_fused_gelu_180 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_181 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
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


cpp_fused__log_softmax_add_nll_loss_forward_182 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (50265L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x1)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        in_out_ptr0[static_cast<long>(x1 + (50265L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (50265L*x0))];
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
                        auto tmp5 = decltype(tmp4)(tmp4 + 50265);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 50265L), "index out of bounds: 0 <= tmp7 < 50265L")
                        auto tmp8 = in_out_ptr0[static_cast<long>(tmp7 + (50265L*x0))];
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg1_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg2_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg3_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (1024, ), (1, ))
    assert_size_stride(arg5_1, (1024, ), (1, ))
    assert_size_stride(arg6_1, (1024, ), (1, ))
    assert_size_stride(arg7_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg8_1, (1024, ), (1, ))
    assert_size_stride(arg9_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg10_1, (1024, ), (1, ))
    assert_size_stride(arg11_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg12_1, (1024, ), (1, ))
    assert_size_stride(arg13_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg14_1, (1024, ), (1, ))
    assert_size_stride(arg15_1, (1024, ), (1, ))
    assert_size_stride(arg16_1, (1024, ), (1, ))
    assert_size_stride(arg17_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg18_1, (4096, ), (1, ))
    assert_size_stride(arg19_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg20_1, (1024, ), (1, ))
    assert_size_stride(arg21_1, (1024, ), (1, ))
    assert_size_stride(arg22_1, (1024, ), (1, ))
    assert_size_stride(arg23_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg24_1, (1024, ), (1, ))
    assert_size_stride(arg25_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg26_1, (1024, ), (1, ))
    assert_size_stride(arg27_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg28_1, (1024, ), (1, ))
    assert_size_stride(arg29_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg30_1, (1024, ), (1, ))
    assert_size_stride(arg31_1, (1024, ), (1, ))
    assert_size_stride(arg32_1, (1024, ), (1, ))
    assert_size_stride(arg33_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg34_1, (4096, ), (1, ))
    assert_size_stride(arg35_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg40_1, (1024, ), (1, ))
    assert_size_stride(arg41_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg42_1, (1024, ), (1, ))
    assert_size_stride(arg43_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (1024, ), (1, ))
    assert_size_stride(arg48_1, (1024, ), (1, ))
    assert_size_stride(arg49_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg50_1, (4096, ), (1, ))
    assert_size_stride(arg51_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg52_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (1024, ), (1, ))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg57_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg63_1, (1024, ), (1, ))
    assert_size_stride(arg64_1, (1024, ), (1, ))
    assert_size_stride(arg65_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg66_1, (4096, ), (1, ))
    assert_size_stride(arg67_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (1024, ), (1, ))
    assert_size_stride(arg70_1, (1024, ), (1, ))
    assert_size_stride(arg71_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg72_1, (1024, ), (1, ))
    assert_size_stride(arg73_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg74_1, (1024, ), (1, ))
    assert_size_stride(arg75_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg78_1, (1024, ), (1, ))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg82_1, (4096, ), (1, ))
    assert_size_stride(arg83_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, ), (1, ))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg87_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg88_1, (1024, ), (1, ))
    assert_size_stride(arg89_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg90_1, (1024, ), (1, ))
    assert_size_stride(arg91_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg94_1, (1024, ), (1, ))
    assert_size_stride(arg95_1, (1024, ), (1, ))
    assert_size_stride(arg96_1, (1024, ), (1, ))
    assert_size_stride(arg97_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg98_1, (4096, ), (1, ))
    assert_size_stride(arg99_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (1024, ), (1, ))
    assert_size_stride(arg103_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg104_1, (1024, ), (1, ))
    assert_size_stride(arg105_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg106_1, (1024, ), (1, ))
    assert_size_stride(arg107_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (1024, ), (1, ))
    assert_size_stride(arg112_1, (1024, ), (1, ))
    assert_size_stride(arg113_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg114_1, (4096, ), (1, ))
    assert_size_stride(arg115_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg120_1, (1024, ), (1, ))
    assert_size_stride(arg121_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg122_1, (1024, ), (1, ))
    assert_size_stride(arg123_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg124_1, (1024, ), (1, ))
    assert_size_stride(arg125_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg126_1, (1024, ), (1, ))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg130_1, (4096, ), (1, ))
    assert_size_stride(arg131_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg142_1, (1024, ), (1, ))
    assert_size_stride(arg143_1, (1024, ), (1, ))
    assert_size_stride(arg144_1, (1024, ), (1, ))
    assert_size_stride(arg145_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg146_1, (4096, ), (1, ))
    assert_size_stride(arg147_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg158_1, (1024, ), (1, ))
    assert_size_stride(arg159_1, (1024, ), (1, ))
    assert_size_stride(arg160_1, (1024, ), (1, ))
    assert_size_stride(arg161_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg162_1, (4096, ), (1, ))
    assert_size_stride(arg163_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (1024, ), (1, ))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg174_1, (1024, ), (1, ))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg178_1, (4096, ), (1, ))
    assert_size_stride(arg179_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg186_1, (1024, ), (1, ))
    assert_size_stride(arg187_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg188_1, (1024, ), (1, ))
    assert_size_stride(arg189_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (1024, ), (1, ))
    assert_size_stride(arg193_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg194_1, (4096, ), (1, ))
    assert_size_stride(arg195_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (1024, ), (1, ))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (1024, ), (1, ))
    assert_size_stride(arg202_1, (1024, ), (1, ))
    assert_size_stride(arg203_1, (1024, ), (1, ))
    assert_size_stride(arg204_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg205_1, (1024, ), (1, ))
    assert_size_stride(arg206_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg207_1, (1024, ), (1, ))
    assert_size_stride(arg208_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg211_1, (1024, ), (1, ))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (1024, ), (1, ))
    assert_size_stride(arg214_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg215_1, (1024, ), (1, ))
    assert_size_stride(arg216_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg217_1, (1024, ), (1, ))
    assert_size_stride(arg218_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg219_1, (1024, ), (1, ))
    assert_size_stride(arg220_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg221_1, (1024, ), (1, ))
    assert_size_stride(arg222_1, (1024, ), (1, ))
    assert_size_stride(arg223_1, (1024, ), (1, ))
    assert_size_stride(arg224_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg225_1, (4096, ), (1, ))
    assert_size_stride(arg226_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (1024, ), (1, ))
    assert_size_stride(arg229_1, (1024, ), (1, ))
    assert_size_stride(arg230_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg231_1, (1024, ), (1, ))
    assert_size_stride(arg232_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg233_1, (1024, ), (1, ))
    assert_size_stride(arg234_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg235_1, (1024, ), (1, ))
    assert_size_stride(arg236_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg237_1, (1024, ), (1, ))
    assert_size_stride(arg238_1, (1024, ), (1, ))
    assert_size_stride(arg239_1, (1024, ), (1, ))
    assert_size_stride(arg240_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg243_1, (1024, ), (1, ))
    assert_size_stride(arg244_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg245_1, (1024, ), (1, ))
    assert_size_stride(arg246_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg247_1, (1024, ), (1, ))
    assert_size_stride(arg248_1, (1024, ), (1, ))
    assert_size_stride(arg249_1, (1024, ), (1, ))
    assert_size_stride(arg250_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg251_1, (4096, ), (1, ))
    assert_size_stride(arg252_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg253_1, (1024, ), (1, ))
    assert_size_stride(arg254_1, (1024, ), (1, ))
    assert_size_stride(arg255_1, (1024, ), (1, ))
    assert_size_stride(arg256_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg257_1, (1024, ), (1, ))
    assert_size_stride(arg258_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg259_1, (1024, ), (1, ))
    assert_size_stride(arg260_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg261_1, (1024, ), (1, ))
    assert_size_stride(arg262_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg263_1, (1024, ), (1, ))
    assert_size_stride(arg264_1, (1024, ), (1, ))
    assert_size_stride(arg265_1, (1024, ), (1, ))
    assert_size_stride(arg266_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg267_1, (1024, ), (1, ))
    assert_size_stride(arg268_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg269_1, (1024, ), (1, ))
    assert_size_stride(arg270_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg271_1, (1024, ), (1, ))
    assert_size_stride(arg272_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg273_1, (1024, ), (1, ))
    assert_size_stride(arg274_1, (1024, ), (1, ))
    assert_size_stride(arg275_1, (1024, ), (1, ))
    assert_size_stride(arg276_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg277_1, (4096, ), (1, ))
    assert_size_stride(arg278_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg279_1, (1024, ), (1, ))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg283_1, (1024, ), (1, ))
    assert_size_stride(arg284_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg285_1, (1024, ), (1, ))
    assert_size_stride(arg286_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg287_1, (1024, ), (1, ))
    assert_size_stride(arg288_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg289_1, (1024, ), (1, ))
    assert_size_stride(arg290_1, (1024, ), (1, ))
    assert_size_stride(arg291_1, (1024, ), (1, ))
    assert_size_stride(arg292_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg293_1, (1024, ), (1, ))
    assert_size_stride(arg294_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg295_1, (1024, ), (1, ))
    assert_size_stride(arg296_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg297_1, (1024, ), (1, ))
    assert_size_stride(arg298_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg299_1, (1024, ), (1, ))
    assert_size_stride(arg300_1, (1024, ), (1, ))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg303_1, (4096, ), (1, ))
    assert_size_stride(arg304_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg305_1, (1024, ), (1, ))
    assert_size_stride(arg306_1, (1024, ), (1, ))
    assert_size_stride(arg307_1, (1024, ), (1, ))
    assert_size_stride(arg308_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg309_1, (1024, ), (1, ))
    assert_size_stride(arg310_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg311_1, (1024, ), (1, ))
    assert_size_stride(arg312_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg313_1, (1024, ), (1, ))
    assert_size_stride(arg314_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg315_1, (1024, ), (1, ))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg319_1, (1024, ), (1, ))
    assert_size_stride(arg320_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg321_1, (1024, ), (1, ))
    assert_size_stride(arg322_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg323_1, (1024, ), (1, ))
    assert_size_stride(arg324_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg325_1, (1024, ), (1, ))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1024, ), (1, ))
    assert_size_stride(arg328_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg329_1, (4096, ), (1, ))
    assert_size_stride(arg330_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg331_1, (1024, ), (1, ))
    assert_size_stride(arg332_1, (1024, ), (1, ))
    assert_size_stride(arg333_1, (1024, ), (1, ))
    assert_size_stride(arg334_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg337_1, (1024, ), (1, ))
    assert_size_stride(arg338_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg339_1, (1024, ), (1, ))
    assert_size_stride(arg340_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg341_1, (1024, ), (1, ))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1024, ), (1, ))
    assert_size_stride(arg344_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg345_1, (1024, ), (1, ))
    assert_size_stride(arg346_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg347_1, (1024, ), (1, ))
    assert_size_stride(arg348_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg349_1, (1024, ), (1, ))
    assert_size_stride(arg350_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg351_1, (1024, ), (1, ))
    assert_size_stride(arg352_1, (1024, ), (1, ))
    assert_size_stride(arg353_1, (1024, ), (1, ))
    assert_size_stride(arg354_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg355_1, (4096, ), (1, ))
    assert_size_stride(arg356_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg357_1, (1024, ), (1, ))
    assert_size_stride(arg358_1, (1024, ), (1, ))
    assert_size_stride(arg359_1, (1024, ), (1, ))
    assert_size_stride(arg360_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg361_1, (1024, ), (1, ))
    assert_size_stride(arg362_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg363_1, (1024, ), (1, ))
    assert_size_stride(arg364_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg365_1, (1024, ), (1, ))
    assert_size_stride(arg366_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg367_1, (1024, ), (1, ))
    assert_size_stride(arg368_1, (1024, ), (1, ))
    assert_size_stride(arg369_1, (1024, ), (1, ))
    assert_size_stride(arg370_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg371_1, (1024, ), (1, ))
    assert_size_stride(arg372_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg373_1, (1024, ), (1, ))
    assert_size_stride(arg374_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg375_1, (1024, ), (1, ))
    assert_size_stride(arg376_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg377_1, (1024, ), (1, ))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (1024, ), (1, ))
    assert_size_stride(arg380_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg381_1, (4096, ), (1, ))
    assert_size_stride(arg382_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg383_1, (1024, ), (1, ))
    assert_size_stride(arg384_1, (1024, ), (1, ))
    assert_size_stride(arg385_1, (1024, ), (1, ))
    assert_size_stride(arg386_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg387_1, (1024, ), (1, ))
    assert_size_stride(arg388_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg389_1, (1024, ), (1, ))
    assert_size_stride(arg390_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg391_1, (1024, ), (1, ))
    assert_size_stride(arg392_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg393_1, (1024, ), (1, ))
    assert_size_stride(arg394_1, (1024, ), (1, ))
    assert_size_stride(arg395_1, (1024, ), (1, ))
    assert_size_stride(arg396_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg397_1, (1024, ), (1, ))
    assert_size_stride(arg398_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg399_1, (1024, ), (1, ))
    assert_size_stride(arg400_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg401_1, (1024, ), (1, ))
    assert_size_stride(arg402_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg403_1, (1024, ), (1, ))
    assert_size_stride(arg404_1, (1024, ), (1, ))
    assert_size_stride(arg405_1, (1024, ), (1, ))
    assert_size_stride(arg406_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg407_1, (4096, ), (1, ))
    assert_size_stride(arg408_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg409_1, (1024, ), (1, ))
    assert_size_stride(arg410_1, (1024, ), (1, ))
    assert_size_stride(arg411_1, (1024, ), (1, ))
    assert_size_stride(arg412_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg413_1, (1024, ), (1, ))
    assert_size_stride(arg414_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg415_1, (1024, ), (1, ))
    assert_size_stride(arg416_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg417_1, (1024, ), (1, ))
    assert_size_stride(arg418_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg419_1, (1024, ), (1, ))
    assert_size_stride(arg420_1, (1024, ), (1, ))
    assert_size_stride(arg421_1, (1024, ), (1, ))
    assert_size_stride(arg422_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg423_1, (1024, ), (1, ))
    assert_size_stride(arg424_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg425_1, (1024, ), (1, ))
    assert_size_stride(arg426_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg427_1, (1024, ), (1, ))
    assert_size_stride(arg428_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg429_1, (1024, ), (1, ))
    assert_size_stride(arg430_1, (1024, ), (1, ))
    assert_size_stride(arg431_1, (1024, ), (1, ))
    assert_size_stride(arg432_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg433_1, (4096, ), (1, ))
    assert_size_stride(arg434_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg435_1, (1024, ), (1, ))
    assert_size_stride(arg436_1, (1024, ), (1, ))
    assert_size_stride(arg437_1, (1024, ), (1, ))
    assert_size_stride(arg438_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg439_1, (1024, ), (1, ))
    assert_size_stride(arg440_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg441_1, (1024, ), (1, ))
    assert_size_stride(arg442_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg443_1, (1024, ), (1, ))
    assert_size_stride(arg444_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg445_1, (1024, ), (1, ))
    assert_size_stride(arg446_1, (1024, ), (1, ))
    assert_size_stride(arg447_1, (1024, ), (1, ))
    assert_size_stride(arg448_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg449_1, (1024, ), (1, ))
    assert_size_stride(arg450_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg451_1, (1024, ), (1, ))
    assert_size_stride(arg452_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg453_1, (1024, ), (1, ))
    assert_size_stride(arg454_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg455_1, (1024, ), (1, ))
    assert_size_stride(arg456_1, (1024, ), (1, ))
    assert_size_stride(arg457_1, (1024, ), (1, ))
    assert_size_stride(arg458_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg459_1, (4096, ), (1, ))
    assert_size_stride(arg460_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg461_1, (1024, ), (1, ))
    assert_size_stride(arg462_1, (1024, ), (1, ))
    assert_size_stride(arg463_1, (1024, ), (1, ))
    assert_size_stride(arg464_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg465_1, (1024, ), (1, ))
    assert_size_stride(arg466_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg467_1, (1024, ), (1, ))
    assert_size_stride(arg468_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg469_1, (1024, ), (1, ))
    assert_size_stride(arg470_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg471_1, (1024, ), (1, ))
    assert_size_stride(arg472_1, (1024, ), (1, ))
    assert_size_stride(arg473_1, (1024, ), (1, ))
    assert_size_stride(arg474_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg475_1, (1024, ), (1, ))
    assert_size_stride(arg476_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg477_1, (1024, ), (1, ))
    assert_size_stride(arg478_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg479_1, (1024, ), (1, ))
    assert_size_stride(arg480_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg481_1, (1024, ), (1, ))
    assert_size_stride(arg482_1, (1024, ), (1, ))
    assert_size_stride(arg483_1, (1024, ), (1, ))
    assert_size_stride(arg484_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg485_1, (4096, ), (1, ))
    assert_size_stride(arg486_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg487_1, (1024, ), (1, ))
    assert_size_stride(arg488_1, (1024, ), (1, ))
    assert_size_stride(arg489_1, (1024, ), (1, ))
    assert_size_stride(arg490_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg491_1, (1024, ), (1, ))
    assert_size_stride(arg492_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg493_1, (1024, ), (1, ))
    assert_size_stride(arg494_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg495_1, (1024, ), (1, ))
    assert_size_stride(arg496_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg497_1, (1024, ), (1, ))
    assert_size_stride(arg498_1, (1024, ), (1, ))
    assert_size_stride(arg499_1, (1024, ), (1, ))
    assert_size_stride(arg500_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg501_1, (1024, ), (1, ))
    assert_size_stride(arg502_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg503_1, (1024, ), (1, ))
    assert_size_stride(arg504_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg505_1, (1024, ), (1, ))
    assert_size_stride(arg506_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg507_1, (1024, ), (1, ))
    assert_size_stride(arg508_1, (1024, ), (1, ))
    assert_size_stride(arg509_1, (1024, ), (1, ))
    assert_size_stride(arg510_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg511_1, (4096, ), (1, ))
    assert_size_stride(arg512_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg513_1, (1024, ), (1, ))
    assert_size_stride(arg514_1, (1024, ), (1, ))
    assert_size_stride(arg515_1, (1024, ), (1, ))
    assert_size_stride(arg516_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg517_1, (1, 50265), (50265, 1))
    assert_size_stride(arg518_1, (1, 1024), (1024, 1))
    assert_size_stride(arg519_1, (1, 1024), (1024, 1))
    buf0 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf7 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mul_native_layer_norm_0(c_void_p(arg519_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg0_1
    del arg2_1
    del arg3_1
    del arg4_1
    del arg519_1
    del arg5_1
    del arg6_1
    buf8 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg7_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf8)
    del arg7_1
    del arg8_1
    buf9 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg9_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf9)
    del arg10_1
    del arg9_1
    buf10 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg11_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf10)
    del arg11_1
    del arg12_1
    buf11 = reinterpret_tensor(buf7, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf7  # reuse
    buf12 = empty((1, 16, 1024, 64), device='cpu', dtype=torch.float32)
    buf13 = empty((1, 16, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_1(c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf14 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf11, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf12, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf13, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf15 = buf14[0]
    del buf14
    buf22 = reinterpret_tensor(buf15, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf15  # reuse
    cpp_fused_clone_2(c_void_p(buf22.data_ptr()))
    buf23 = reinterpret_tensor(buf13, (1024, 1024), (1024, 1), 0); del buf13  # reuse
    # Source Nodes: [hidden_states_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf22, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg13_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf23)
    del arg13_1
    del arg14_1
    buf24 = buf5; del buf5  # reuse
    buf25 = buf4; del buf4  # reuse
    buf27 = reinterpret_tensor(buf22, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf22  # reuse
    cpp_fused_add_native_layer_norm_3(c_void_p(buf3.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()))
    del arg15_1
    del arg16_1
    buf28 = empty((1024, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf27, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg17_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf28)
    del arg17_1
    del arg18_1
    buf29 = reinterpret_tensor(buf28, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf28  # reuse
    cpp_fused_gelu_4(c_void_p(buf29.data_ptr()))
    buf30 = reinterpret_tensor(buf27, (1024, 1024), (1024, 1), 0); del buf27  # reuse
    # Source Nodes: [hidden_states_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf29, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg19_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf30)
    del arg19_1
    del arg20_1
    buf31 = buf25; del buf25  # reuse
    buf32 = buf24; del buf24  # reuse
    buf34 = reinterpret_tensor(buf12, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf12  # reuse
    cpp_fused_add_native_layer_norm_5(c_void_p(buf3.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()))
    del arg21_1
    del arg22_1
    buf35 = reinterpret_tensor(buf11, (1024, 1024), (1024, 1), 0); del buf11  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg24_1, reinterpret_tensor(buf34, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg23_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf35)
    del arg23_1
    del arg24_1
    buf36 = buf9; del buf9  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg26_1, reinterpret_tensor(buf34, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg25_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf36)
    del arg25_1
    del arg26_1
    buf37 = buf8; del buf8  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg28_1, reinterpret_tensor(buf34, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg27_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf37)
    del arg27_1
    del arg28_1
    buf38 = reinterpret_tensor(buf34, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf34  # reuse
    buf39 = reinterpret_tensor(buf10, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf10  # reuse
    buf40 = empty((1, 16, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_6(c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    del buf35
    del buf36
    # Source Nodes: [], Original ATen: []
    buf41 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf38, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf39, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf40, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf42 = buf41[0]
    del buf41
    buf49 = reinterpret_tensor(buf42, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf42  # reuse
    cpp_fused_clone_7(c_void_p(buf49.data_ptr()))
    buf50 = reinterpret_tensor(buf40, (1024, 1024), (1024, 1), 0); del buf40  # reuse
    # Source Nodes: [hidden_states_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg30_1, reinterpret_tensor(buf49, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg29_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf50)
    del arg29_1
    del arg30_1
    buf51 = buf32; del buf32  # reuse
    buf52 = buf31; del buf31  # reuse
    buf54 = reinterpret_tensor(buf49, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf49  # reuse
    cpp_fused_add_native_layer_norm_8(c_void_p(buf3.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf54.data_ptr()))
    del arg31_1
    del arg32_1
    buf55 = reinterpret_tensor(buf29, (1024, 4096), (4096, 1), 0); del buf29  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg34_1, reinterpret_tensor(buf54, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg33_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf55)
    del arg33_1
    del arg34_1
    buf56 = reinterpret_tensor(buf55, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf55  # reuse
    cpp_fused_gelu_9(c_void_p(buf56.data_ptr()))
    buf57 = reinterpret_tensor(buf54, (1024, 1024), (1024, 1), 0); del buf54  # reuse
    # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg36_1, reinterpret_tensor(buf56, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg35_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf57)
    del arg35_1
    del arg36_1
    buf58 = reinterpret_tensor(buf57, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf57  # reuse
    buf59 = buf52; del buf52  # reuse
    buf60 = buf51; del buf51  # reuse
    buf62 = reinterpret_tensor(buf39, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf39  # reuse
    cpp_fused_add_native_layer_norm_10(c_void_p(buf58.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf62.data_ptr()))
    del arg37_1
    del arg38_1
    buf63 = buf50; del buf50  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_2_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg40_1, reinterpret_tensor(buf62, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg39_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf63)
    del arg39_1
    del arg40_1
    buf64 = buf30; del buf30  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_2_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg42_1, reinterpret_tensor(buf62, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg41_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf64)
    del arg41_1
    del arg42_1
    buf65 = reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0); del buf3  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_2_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg44_1, reinterpret_tensor(buf62, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg43_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf65)
    del arg43_1
    del arg44_1
    buf66 = reinterpret_tensor(buf62, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf62  # reuse
    buf67 = reinterpret_tensor(buf23, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf23  # reuse
    buf68 = buf38; del buf38  # reuse
    cpp_fused_clone_11(c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf69 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf66, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf67, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf68, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf70 = buf69[0]
    del buf69
    buf77 = reinterpret_tensor(buf70, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf70  # reuse
    cpp_fused_clone_12(c_void_p(buf77.data_ptr()))
    buf78 = reinterpret_tensor(buf68, (1024, 1024), (1024, 1), 0); del buf68  # reuse
    # Source Nodes: [hidden_states_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg46_1, reinterpret_tensor(buf77, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg45_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf78)
    del arg45_1
    del arg46_1
    buf79 = buf60; del buf60  # reuse
    buf80 = buf59; del buf59  # reuse
    buf82 = reinterpret_tensor(buf77, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf77  # reuse
    cpp_fused_add_native_layer_norm_13(c_void_p(buf58.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()))
    del arg47_1
    del arg48_1
    buf83 = reinterpret_tensor(buf56, (1024, 4096), (4096, 1), 0); del buf56  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_2_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg50_1, reinterpret_tensor(buf82, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg49_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf83)
    del arg49_1
    del arg50_1
    buf84 = reinterpret_tensor(buf83, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf83  # reuse
    cpp_fused_gelu_14(c_void_p(buf84.data_ptr()))
    buf85 = reinterpret_tensor(buf82, (1024, 1024), (1024, 1), 0); del buf82  # reuse
    # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg52_1, reinterpret_tensor(buf84, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg51_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf85)
    del arg51_1
    del arg52_1
    buf86 = buf80; del buf80  # reuse
    buf87 = buf79; del buf79  # reuse
    buf89 = reinterpret_tensor(buf67, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf67  # reuse
    cpp_fused_add_native_layer_norm_15(c_void_p(buf58.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()))
    del arg53_1
    del arg54_1
    buf90 = reinterpret_tensor(buf66, (1024, 1024), (1024, 1), 0); del buf66  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg56_1, reinterpret_tensor(buf89, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg55_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf90)
    del arg55_1
    del arg56_1
    buf91 = buf65; del buf65  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg58_1, reinterpret_tensor(buf89, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf91)
    del arg57_1
    del arg58_1
    buf92 = buf64; del buf64  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg60_1, reinterpret_tensor(buf89, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg59_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf92)
    del arg59_1
    del arg60_1
    buf93 = reinterpret_tensor(buf89, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf89  # reuse
    buf94 = reinterpret_tensor(buf63, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf63  # reuse
    buf95 = reinterpret_tensor(buf37, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf37  # reuse
    cpp_fused_clone_16(c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    del buf90
    del buf91
    # Source Nodes: [], Original ATen: []
    buf96 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf93, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf94, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf95, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf97 = buf96[0]
    del buf96
    buf104 = reinterpret_tensor(buf97, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf97  # reuse
    cpp_fused_clone_17(c_void_p(buf104.data_ptr()))
    buf105 = reinterpret_tensor(buf95, (1024, 1024), (1024, 1), 0); del buf95  # reuse
    # Source Nodes: [hidden_states_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg62_1, reinterpret_tensor(buf104, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg61_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf105)
    del arg61_1
    del arg62_1
    buf106 = buf87; del buf87  # reuse
    buf107 = buf86; del buf86  # reuse
    buf109 = reinterpret_tensor(buf104, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf104  # reuse
    cpp_fused_add_native_layer_norm_18(c_void_p(buf58.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()))
    del arg63_1
    del arg64_1
    buf110 = reinterpret_tensor(buf84, (1024, 4096), (4096, 1), 0); del buf84  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg66_1, reinterpret_tensor(buf109, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg65_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf110)
    del arg65_1
    del arg66_1
    buf111 = reinterpret_tensor(buf110, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf110  # reuse
    cpp_fused_gelu_19(c_void_p(buf111.data_ptr()))
    buf112 = reinterpret_tensor(buf109, (1024, 1024), (1024, 1), 0); del buf109  # reuse
    # Source Nodes: [hidden_states_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg68_1, reinterpret_tensor(buf111, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg67_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf112)
    del arg67_1
    del arg68_1
    buf113 = reinterpret_tensor(buf112, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf112  # reuse
    buf114 = buf107; del buf107  # reuse
    buf115 = buf106; del buf106  # reuse
    buf117 = reinterpret_tensor(buf94, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf94  # reuse
    cpp_fused_add_native_layer_norm_20(c_void_p(buf113.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()))
    del arg69_1
    del arg70_1
    buf118 = buf85; del buf85  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_4_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg72_1, reinterpret_tensor(buf117, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf118)
    del arg71_1
    del arg72_1
    buf119 = buf78; del buf78  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_4_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg74_1, reinterpret_tensor(buf117, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf119)
    del arg73_1
    del arg74_1
    buf120 = reinterpret_tensor(buf58, (1024, 1024), (1024, 1), 0); del buf58  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_4_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg76_1, reinterpret_tensor(buf117, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg75_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf120)
    del arg75_1
    del arg76_1
    buf121 = reinterpret_tensor(buf117, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf117  # reuse
    buf122 = reinterpret_tensor(buf105, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf105  # reuse
    buf123 = buf93; del buf93  # reuse
    cpp_fused_clone_21(c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf124 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf121, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf122, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf123, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf125 = buf124[0]
    del buf124
    buf132 = reinterpret_tensor(buf125, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf125  # reuse
    cpp_fused_clone_22(c_void_p(buf132.data_ptr()))
    buf133 = reinterpret_tensor(buf123, (1024, 1024), (1024, 1), 0); del buf123  # reuse
    # Source Nodes: [hidden_states_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg78_1, reinterpret_tensor(buf132, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg77_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf133)
    del arg77_1
    del arg78_1
    buf134 = buf115; del buf115  # reuse
    buf135 = buf114; del buf114  # reuse
    buf137 = reinterpret_tensor(buf132, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf132  # reuse
    cpp_fused_add_native_layer_norm_23(c_void_p(buf113.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf137.data_ptr()))
    del arg79_1
    del arg80_1
    buf138 = reinterpret_tensor(buf111, (1024, 4096), (4096, 1), 0); del buf111  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_4_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg82_1, reinterpret_tensor(buf137, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg81_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf138)
    del arg81_1
    del arg82_1
    buf139 = reinterpret_tensor(buf138, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf138  # reuse
    cpp_fused_gelu_24(c_void_p(buf139.data_ptr()))
    buf140 = reinterpret_tensor(buf137, (1024, 1024), (1024, 1), 0); del buf137  # reuse
    # Source Nodes: [hidden_states_54], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg84_1, reinterpret_tensor(buf139, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg83_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf140)
    del arg83_1
    del arg84_1
    buf141 = buf135; del buf135  # reuse
    buf142 = buf134; del buf134  # reuse
    buf144 = reinterpret_tensor(buf122, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf122  # reuse
    cpp_fused_add_native_layer_norm_25(c_void_p(buf113.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()))
    del arg85_1
    del arg86_1
    buf145 = reinterpret_tensor(buf121, (1024, 1024), (1024, 1), 0); del buf121  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg88_1, reinterpret_tensor(buf144, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg87_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf145)
    del arg87_1
    del arg88_1
    buf146 = buf120; del buf120  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg90_1, reinterpret_tensor(buf144, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg89_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf146)
    del arg89_1
    del arg90_1
    buf147 = buf119; del buf119  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg92_1, reinterpret_tensor(buf144, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg91_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf147)
    del arg91_1
    del arg92_1
    buf148 = reinterpret_tensor(buf144, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf144  # reuse
    buf149 = reinterpret_tensor(buf118, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf118  # reuse
    buf150 = reinterpret_tensor(buf92, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf92  # reuse
    cpp_fused_clone_26(c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    del buf145
    del buf146
    # Source Nodes: [], Original ATen: []
    buf151 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf148, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf149, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf150, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf152 = buf151[0]
    del buf151
    buf159 = reinterpret_tensor(buf152, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf152  # reuse
    cpp_fused_clone_27(c_void_p(buf159.data_ptr()))
    buf160 = reinterpret_tensor(buf150, (1024, 1024), (1024, 1), 0); del buf150  # reuse
    # Source Nodes: [hidden_states_59], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg94_1, reinterpret_tensor(buf159, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg93_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf160)
    del arg93_1
    del arg94_1
    buf161 = buf142; del buf142  # reuse
    buf162 = buf141; del buf141  # reuse
    buf164 = reinterpret_tensor(buf159, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf159  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf113.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()))
    del arg95_1
    del arg96_1
    buf165 = reinterpret_tensor(buf139, (1024, 4096), (4096, 1), 0); del buf139  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg98_1, reinterpret_tensor(buf164, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg97_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf165)
    del arg97_1
    del arg98_1
    buf166 = reinterpret_tensor(buf165, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf165  # reuse
    cpp_fused_gelu_29(c_void_p(buf166.data_ptr()))
    buf167 = reinterpret_tensor(buf164, (1024, 1024), (1024, 1), 0); del buf164  # reuse
    # Source Nodes: [hidden_states_65], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg100_1, reinterpret_tensor(buf166, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg99_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf167)
    del arg100_1
    del arg99_1
    buf168 = reinterpret_tensor(buf167, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf167  # reuse
    buf169 = buf162; del buf162  # reuse
    buf170 = buf161; del buf161  # reuse
    buf172 = reinterpret_tensor(buf149, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf149  # reuse
    cpp_fused_add_native_layer_norm_30(c_void_p(buf168.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()))
    del arg101_1
    del arg102_1
    buf173 = buf160; del buf160  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_6_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg104_1, reinterpret_tensor(buf172, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg103_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf173)
    del arg103_1
    del arg104_1
    buf174 = buf140; del buf140  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_6_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg106_1, reinterpret_tensor(buf172, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg105_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf174)
    del arg105_1
    del arg106_1
    buf175 = buf133; del buf133  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_6_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg108_1, reinterpret_tensor(buf172, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg107_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf175)
    del arg107_1
    del arg108_1
    buf176 = reinterpret_tensor(buf172, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf172  # reuse
    buf177 = reinterpret_tensor(buf113, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf113  # reuse
    buf178 = buf148; del buf148  # reuse
    cpp_fused_clone_31(c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf179 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf176, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf177, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf178, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf180 = buf179[0]
    del buf179
    buf187 = reinterpret_tensor(buf180, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf180  # reuse
    cpp_fused_clone_32(c_void_p(buf187.data_ptr()))
    buf188 = reinterpret_tensor(buf178, (1024, 1024), (1024, 1), 0); del buf178  # reuse
    # Source Nodes: [hidden_states_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg110_1, reinterpret_tensor(buf187, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg109_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf188)
    del arg109_1
    del arg110_1
    buf189 = buf170; del buf170  # reuse
    buf190 = buf169; del buf169  # reuse
    buf192 = reinterpret_tensor(buf187, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf187  # reuse
    cpp_fused_add_native_layer_norm_33(c_void_p(buf168.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf192.data_ptr()))
    del arg111_1
    del arg112_1
    buf193 = reinterpret_tensor(buf166, (1024, 4096), (4096, 1), 0); del buf166  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_6_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg114_1, reinterpret_tensor(buf192, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg113_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf193)
    del arg113_1
    del arg114_1
    buf194 = reinterpret_tensor(buf193, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf193  # reuse
    cpp_fused_gelu_34(c_void_p(buf194.data_ptr()))
    buf195 = reinterpret_tensor(buf192, (1024, 1024), (1024, 1), 0); del buf192  # reuse
    # Source Nodes: [hidden_states_76], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg116_1, reinterpret_tensor(buf194, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg115_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf195)
    del arg115_1
    del arg116_1
    buf196 = buf190; del buf190  # reuse
    buf197 = buf189; del buf189  # reuse
    buf199 = reinterpret_tensor(buf177, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf177  # reuse
    cpp_fused_add_native_layer_norm_35(c_void_p(buf168.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()))
    del arg117_1
    del arg118_1
    buf200 = reinterpret_tensor(buf176, (1024, 1024), (1024, 1), 0); del buf176  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_7_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg120_1, reinterpret_tensor(buf199, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg119_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf200)
    del arg119_1
    del arg120_1
    buf201 = buf175; del buf175  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_7_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg122_1, reinterpret_tensor(buf199, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg121_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf201)
    del arg121_1
    del arg122_1
    buf202 = buf174; del buf174  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_7_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg124_1, reinterpret_tensor(buf199, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg123_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf202)
    del arg123_1
    del arg124_1
    buf203 = reinterpret_tensor(buf199, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf199  # reuse
    buf204 = reinterpret_tensor(buf173, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf173  # reuse
    buf205 = reinterpret_tensor(buf147, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf147  # reuse
    cpp_fused_clone_36(c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()))
    del buf200
    del buf201
    # Source Nodes: [], Original ATen: []
    buf206 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf203, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf204, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf205, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf207 = buf206[0]
    del buf206
    buf214 = reinterpret_tensor(buf207, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf207  # reuse
    cpp_fused_clone_37(c_void_p(buf214.data_ptr()))
    buf215 = reinterpret_tensor(buf205, (1024, 1024), (1024, 1), 0); del buf205  # reuse
    # Source Nodes: [hidden_states_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg126_1, reinterpret_tensor(buf214, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg125_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf215)
    del arg125_1
    del arg126_1
    buf216 = buf197; del buf197  # reuse
    buf217 = buf196; del buf196  # reuse
    buf219 = reinterpret_tensor(buf214, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf214  # reuse
    cpp_fused_add_native_layer_norm_38(c_void_p(buf168.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()))
    del arg127_1
    del arg128_1
    buf220 = reinterpret_tensor(buf194, (1024, 4096), (4096, 1), 0); del buf194  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_7_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg130_1, reinterpret_tensor(buf219, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg129_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf220)
    del arg129_1
    del arg130_1
    buf221 = reinterpret_tensor(buf220, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf220  # reuse
    cpp_fused_gelu_39(c_void_p(buf221.data_ptr()))
    buf222 = reinterpret_tensor(buf219, (1024, 1024), (1024, 1), 0); del buf219  # reuse
    # Source Nodes: [hidden_states_87], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg132_1, reinterpret_tensor(buf221, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg131_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf222)
    del arg131_1
    del arg132_1
    buf223 = reinterpret_tensor(buf222, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf222  # reuse
    buf224 = buf217; del buf217  # reuse
    buf225 = buf216; del buf216  # reuse
    buf227 = reinterpret_tensor(buf204, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf204  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf223.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf227.data_ptr()))
    del arg133_1
    del arg134_1
    buf228 = buf215; del buf215  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_8_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg136_1, reinterpret_tensor(buf227, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg135_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf228)
    del arg135_1
    del arg136_1
    buf229 = buf195; del buf195  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_8_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf227, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf229)
    del arg137_1
    del arg138_1
    buf230 = buf188; del buf188  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_8_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg140_1, reinterpret_tensor(buf227, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg139_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf230)
    del arg139_1
    del arg140_1
    buf231 = reinterpret_tensor(buf227, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf227  # reuse
    buf232 = reinterpret_tensor(buf168, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf168  # reuse
    buf233 = buf203; del buf203  # reuse
    cpp_fused_clone_41(c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf234 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf231, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf232, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf233, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf235 = buf234[0]
    del buf234
    buf242 = reinterpret_tensor(buf235, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf235  # reuse
    cpp_fused_clone_42(c_void_p(buf242.data_ptr()))
    buf243 = reinterpret_tensor(buf233, (1024, 1024), (1024, 1), 0); del buf233  # reuse
    # Source Nodes: [hidden_states_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg142_1, reinterpret_tensor(buf242, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg141_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf243)
    del arg141_1
    del arg142_1
    buf244 = buf225; del buf225  # reuse
    buf245 = buf224; del buf224  # reuse
    buf247 = reinterpret_tensor(buf242, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf242  # reuse
    cpp_fused_add_native_layer_norm_43(c_void_p(buf223.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()))
    del arg143_1
    del arg144_1
    buf248 = reinterpret_tensor(buf221, (1024, 4096), (4096, 1), 0); del buf221  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_8_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg146_1, reinterpret_tensor(buf247, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg145_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf248)
    del arg145_1
    del arg146_1
    buf249 = reinterpret_tensor(buf248, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf248  # reuse
    cpp_fused_gelu_44(c_void_p(buf249.data_ptr()))
    buf250 = reinterpret_tensor(buf247, (1024, 1024), (1024, 1), 0); del buf247  # reuse
    # Source Nodes: [hidden_states_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg148_1, reinterpret_tensor(buf249, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg147_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf250)
    del arg147_1
    del arg148_1
    buf251 = buf245; del buf245  # reuse
    buf252 = buf244; del buf244  # reuse
    buf254 = reinterpret_tensor(buf232, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf232  # reuse
    cpp_fused_add_native_layer_norm_45(c_void_p(buf223.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf254.data_ptr()))
    del arg149_1
    del arg150_1
    buf255 = reinterpret_tensor(buf231, (1024, 1024), (1024, 1), 0); del buf231  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_9_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg152_1, reinterpret_tensor(buf254, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg151_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf255)
    del arg151_1
    del arg152_1
    buf256 = buf230; del buf230  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_9_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg154_1, reinterpret_tensor(buf254, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg153_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf256)
    del arg153_1
    del arg154_1
    buf257 = buf229; del buf229  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_9_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg156_1, reinterpret_tensor(buf254, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg155_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf257)
    del arg155_1
    del arg156_1
    buf258 = reinterpret_tensor(buf254, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf254  # reuse
    buf259 = reinterpret_tensor(buf228, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf228  # reuse
    buf260 = reinterpret_tensor(buf202, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf202  # reuse
    cpp_fused_clone_46(c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    del buf255
    del buf256
    # Source Nodes: [], Original ATen: []
    buf261 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf258, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf259, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf260, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf262 = buf261[0]
    del buf261
    buf269 = reinterpret_tensor(buf262, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf262  # reuse
    cpp_fused_clone_47(c_void_p(buf269.data_ptr()))
    buf270 = reinterpret_tensor(buf260, (1024, 1024), (1024, 1), 0); del buf260  # reuse
    # Source Nodes: [hidden_states_103], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg158_1, reinterpret_tensor(buf269, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg157_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf270)
    del arg157_1
    del arg158_1
    buf271 = buf252; del buf252  # reuse
    buf272 = buf251; del buf251  # reuse
    buf274 = reinterpret_tensor(buf269, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf269  # reuse
    cpp_fused_add_native_layer_norm_48(c_void_p(buf223.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf274.data_ptr()))
    del arg159_1
    del arg160_1
    buf275 = reinterpret_tensor(buf249, (1024, 4096), (4096, 1), 0); del buf249  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_9_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg162_1, reinterpret_tensor(buf274, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg161_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf275)
    del arg161_1
    del arg162_1
    buf276 = reinterpret_tensor(buf275, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf275  # reuse
    cpp_fused_gelu_49(c_void_p(buf276.data_ptr()))
    buf277 = reinterpret_tensor(buf274, (1024, 1024), (1024, 1), 0); del buf274  # reuse
    # Source Nodes: [hidden_states_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg164_1, reinterpret_tensor(buf276, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg163_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf277)
    del arg163_1
    del arg164_1
    buf278 = reinterpret_tensor(buf277, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf277  # reuse
    buf279 = buf272; del buf272  # reuse
    buf280 = buf271; del buf271  # reuse
    buf282 = reinterpret_tensor(buf259, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf259  # reuse
    cpp_fused_add_native_layer_norm_50(c_void_p(buf278.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf282.data_ptr()))
    del arg165_1
    del arg166_1
    buf283 = buf270; del buf270  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_10_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg168_1, reinterpret_tensor(buf282, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg167_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf283)
    del arg167_1
    del arg168_1
    buf284 = buf250; del buf250  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_10_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg170_1, reinterpret_tensor(buf282, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg169_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf284)
    del arg169_1
    del arg170_1
    buf285 = buf243; del buf243  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_10_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg172_1, reinterpret_tensor(buf282, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf285)
    del arg171_1
    del arg172_1
    buf286 = reinterpret_tensor(buf282, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf282  # reuse
    buf287 = reinterpret_tensor(buf223, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf223  # reuse
    buf288 = buf258; del buf258  # reuse
    cpp_fused_clone_51(c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf289 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf286, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf287, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf288, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf290 = buf289[0]
    del buf289
    buf297 = reinterpret_tensor(buf290, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf290  # reuse
    cpp_fused_clone_52(c_void_p(buf297.data_ptr()))
    buf298 = reinterpret_tensor(buf288, (1024, 1024), (1024, 1), 0); del buf288  # reuse
    # Source Nodes: [hidden_states_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg174_1, reinterpret_tensor(buf297, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg173_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf298)
    del arg173_1
    del arg174_1
    buf299 = buf280; del buf280  # reuse
    buf300 = buf279; del buf279  # reuse
    buf302 = reinterpret_tensor(buf297, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf297  # reuse
    cpp_fused_add_native_layer_norm_53(c_void_p(buf278.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf302.data_ptr()))
    del arg175_1
    del arg176_1
    buf303 = reinterpret_tensor(buf276, (1024, 4096), (4096, 1), 0); del buf276  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_10_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg178_1, reinterpret_tensor(buf302, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg177_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf303)
    del arg177_1
    del arg178_1
    buf304 = reinterpret_tensor(buf303, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf303  # reuse
    cpp_fused_gelu_54(c_void_p(buf304.data_ptr()))
    buf305 = reinterpret_tensor(buf302, (1024, 1024), (1024, 1), 0); del buf302  # reuse
    # Source Nodes: [hidden_states_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg180_1, reinterpret_tensor(buf304, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg179_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf305)
    del arg179_1
    del arg180_1
    buf306 = buf300; del buf300  # reuse
    buf307 = buf299; del buf299  # reuse
    buf309 = reinterpret_tensor(buf287, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf287  # reuse
    cpp_fused_add_native_layer_norm_55(c_void_p(buf278.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf309.data_ptr()))
    del arg181_1
    del arg182_1
    buf310 = reinterpret_tensor(buf286, (1024, 1024), (1024, 1), 0); del buf286  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_11_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg184_1, reinterpret_tensor(buf309, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg183_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf310)
    del arg183_1
    del arg184_1
    buf311 = buf285; del buf285  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_11_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg186_1, reinterpret_tensor(buf309, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg185_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf311)
    del arg185_1
    del arg186_1
    buf312 = buf284; del buf284  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_11_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg188_1, reinterpret_tensor(buf309, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg187_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf312)
    del arg187_1
    del arg188_1
    buf313 = reinterpret_tensor(buf309, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf309  # reuse
    buf314 = reinterpret_tensor(buf283, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf283  # reuse
    buf315 = reinterpret_tensor(buf257, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf257  # reuse
    cpp_fused_clone_56(c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()))
    del buf310
    # Source Nodes: [], Original ATen: []
    buf316 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf313, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf314, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf315, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf317 = buf316[0]
    del buf316
    buf324 = reinterpret_tensor(buf317, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf317  # reuse
    cpp_fused_clone_57(c_void_p(buf324.data_ptr()))
    buf325 = reinterpret_tensor(buf315, (1024, 1024), (1024, 1), 0); del buf315  # reuse
    # Source Nodes: [hidden_states_125], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg190_1, reinterpret_tensor(buf324, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg189_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf325)
    del arg189_1
    del arg190_1
    buf326 = buf307; del buf307  # reuse
    buf327 = buf306; del buf306  # reuse
    buf329 = reinterpret_tensor(buf324, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf324  # reuse
    cpp_fused_add_native_layer_norm_58(c_void_p(buf278.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf329.data_ptr()))
    del arg191_1
    del arg192_1
    buf330 = reinterpret_tensor(buf304, (1024, 4096), (4096, 1), 0); del buf304  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_11_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg194_1, reinterpret_tensor(buf329, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg193_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf330)
    del arg193_1
    del arg194_1
    buf331 = reinterpret_tensor(buf330, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf330  # reuse
    cpp_fused_gelu_59(c_void_p(buf331.data_ptr()))
    buf332 = reinterpret_tensor(buf329, (1024, 1024), (1024, 1), 0); del buf329  # reuse
    # Source Nodes: [hidden_states_131], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg196_1, reinterpret_tensor(buf331, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg195_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf332)
    del arg195_1
    del arg196_1
    buf333 = reinterpret_tensor(buf332, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf332  # reuse
    buf334 = buf327; del buf327  # reuse
    buf335 = buf326; del buf326  # reuse
    buf337 = empty((1, ), device='cpu', dtype=torch.int64)
    buf338 = reinterpret_tensor(buf314, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf314  # reuse
    buf339 = buf1; del buf1  # reuse
    buf340 = buf0; del buf0  # reuse
    buf342 = reinterpret_tensor(buf313, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf313  # reuse
    buf343 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf344 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf346 = reinterpret_tensor(buf312, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf312  # reuse
    cpp_fused_add_clone_copy_embedding_eq_fill_masked_fill_mul_native_layer_norm_ne_select_scatter_slice_scatter_sum_view_60(c_void_p(buf333.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(arg518_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf346.data_ptr()))
    del arg199_1
    del arg1_1
    del arg200_1
    del arg201_1
    del arg202_1
    del arg203_1
    del buf339
    del buf340
    buf347 = reinterpret_tensor(buf338, (1024, 1024), (1024, 1), 0); del buf338  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg205_1, reinterpret_tensor(buf346, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg204_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf347)
    del arg204_1
    del arg205_1
    buf348 = buf325; del buf325  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg207_1, reinterpret_tensor(buf346, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg206_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf348)
    del arg206_1
    del arg207_1
    buf349 = reinterpret_tensor(buf305, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf305  # reuse
    buf350 = reinterpret_tensor(buf298, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf298  # reuse
    cpp_fused_clone_61(c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()))
    buf351 = empty((16, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf349, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf350, (16, 64, 1024), (65536, 1, 64), 0), out=buf351)
    buf352 = empty_strided((16, 1024, 1), (1024, 1, 16384), device='cpu', dtype=torch.float32)
    buf353 = buf351; del buf351  # reuse
    buf354 = empty_strided((16, 1024, 1), (1024, 1, 16384), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_62(c_void_p(buf353.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf354.data_ptr()))
    buf355 = reinterpret_tensor(buf350, (1024, 1024), (1024, 1), 0); del buf350  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg209_1, reinterpret_tensor(buf346, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg208_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf355)
    del arg208_1
    del arg209_1
    buf356 = buf353; del buf353  # reuse
    buf357 = reinterpret_tensor(buf346, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf346  # reuse
    cpp_fused__softmax_clone_63(c_void_p(buf356.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf357.data_ptr()))
    buf358 = reinterpret_tensor(buf355, (16, 1024, 64), (65536, 64, 1), 0); del buf355  # reuse
    # Source Nodes: [attn_output_60, attn_weights_27], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf356, reinterpret_tensor(buf357, (16, 1024, 64), (65536, 64, 1), 0), out=buf358)
    buf359 = reinterpret_tensor(buf357, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf357  # reuse
    cpp_fused_clone_64(c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()))
    buf360 = reinterpret_tensor(buf358, (1024, 1024), (1024, 1), 0); del buf358  # reuse
    # Source Nodes: [hidden_states_140], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg211_1, reinterpret_tensor(buf359, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg210_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf360)
    del arg210_1
    del arg211_1
    buf361 = buf344; del buf344  # reuse
    buf362 = buf343; del buf343  # reuse
    buf364 = reinterpret_tensor(buf359, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf359  # reuse
    cpp_fused_add_native_layer_norm_65(c_void_p(buf342.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf364.data_ptr()))
    del arg212_1
    del arg213_1
    del buf361
    del buf362
    buf365 = reinterpret_tensor(buf349, (1024, 1024), (1024, 1), 0); del buf349  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg215_1, reinterpret_tensor(buf364, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg214_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf365)
    del arg214_1
    del arg215_1
    buf366 = buf364; del buf364  # reuse
    cpp_fused_native_layer_norm_66(c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(buf366.data_ptr()))
    del arg197_1
    del arg198_1
    buf367 = reinterpret_tensor(buf333, (1024, 1024), (1024, 1), 0); del buf333  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg217_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg216_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf367)
    del arg216_1
    del arg217_1
    buf368 = buf348; del buf348  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg219_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg218_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf368)
    del arg218_1
    del arg219_1
    buf369 = reinterpret_tensor(buf347, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf347  # reuse
    buf370 = reinterpret_tensor(buf278, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf278  # reuse
    buf371 = reinterpret_tensor(buf311, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf311  # reuse
    cpp_fused_clone_67(c_void_p(buf365.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf372 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf369, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf370, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf371, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf373 = buf372[0]
    del buf372
    buf380 = reinterpret_tensor(buf373, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf373  # reuse
    cpp_fused_clone_68(c_void_p(buf380.data_ptr()))
    buf381 = reinterpret_tensor(buf371, (1024, 1024), (1024, 1), 0); del buf371  # reuse
    # Source Nodes: [hidden_states_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg221_1, reinterpret_tensor(buf380, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg220_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf381)
    del arg220_1
    del arg221_1
    buf382 = buf335; del buf335  # reuse
    buf383 = buf334; del buf334  # reuse
    buf385 = reinterpret_tensor(buf380, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf380  # reuse
    cpp_fused_add_native_layer_norm_69(c_void_p(buf342.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf385.data_ptr()))
    del arg222_1
    del arg223_1
    buf386 = reinterpret_tensor(buf331, (1024, 4096), (4096, 1), 0); del buf331  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg225_1, reinterpret_tensor(buf385, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg224_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf386)
    del arg224_1
    del arg225_1
    buf387 = reinterpret_tensor(buf386, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf386  # reuse
    cpp_fused_gelu_70(c_void_p(buf387.data_ptr()))
    buf388 = reinterpret_tensor(buf385, (1024, 1024), (1024, 1), 0); del buf385  # reuse
    # Source Nodes: [hidden_states_150], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg227_1, reinterpret_tensor(buf387, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg226_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf388)
    del arg226_1
    del arg227_1
    buf389 = buf383; del buf383  # reuse
    buf390 = buf382; del buf382  # reuse
    buf392 = reinterpret_tensor(buf370, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf370  # reuse
    cpp_fused_add_native_layer_norm_71(c_void_p(buf342.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf392.data_ptr()))
    del arg228_1
    del arg229_1
    buf393 = reinterpret_tensor(buf369, (1024, 1024), (1024, 1), 0); del buf369  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg231_1, reinterpret_tensor(buf392, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg230_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf393)
    del arg230_1
    del arg231_1
    buf394 = buf368; del buf368  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg233_1, reinterpret_tensor(buf392, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg232_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf394)
    del arg232_1
    del arg233_1
    buf395 = reinterpret_tensor(buf367, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf367  # reuse
    buf396 = reinterpret_tensor(buf365, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf365  # reuse
    cpp_fused_clone_72(c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    buf397 = buf356; del buf356  # reuse
    # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf395, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf396, (16, 64, 1024), (65536, 1, 64), 0), out=buf397)
    buf398 = buf354; del buf354  # reuse
    buf399 = buf397; del buf397  # reuse
    buf400 = buf352; del buf352  # reuse
    cpp_fused__softmax_73(c_void_p(buf399.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf400.data_ptr()))
    buf401 = reinterpret_tensor(buf396, (1024, 1024), (1024, 1), 0); del buf396  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg235_1, reinterpret_tensor(buf392, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg234_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf401)
    del arg234_1
    del arg235_1
    buf402 = buf399; del buf399  # reuse
    buf403 = reinterpret_tensor(buf392, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf392  # reuse
    cpp_fused__softmax_clone_74(c_void_p(buf402.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf403.data_ptr()))
    buf404 = reinterpret_tensor(buf401, (16, 1024, 64), (65536, 64, 1), 0); del buf401  # reuse
    # Source Nodes: [attn_output_70, attn_weights_33], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf402, reinterpret_tensor(buf403, (16, 1024, 64), (65536, 64, 1), 0), out=buf404)
    buf405 = reinterpret_tensor(buf403, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf403  # reuse
    cpp_fused_clone_75(c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()))
    buf406 = reinterpret_tensor(buf404, (1024, 1024), (1024, 1), 0); del buf404  # reuse
    # Source Nodes: [hidden_states_155], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg237_1, reinterpret_tensor(buf405, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg236_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf406)
    del arg236_1
    del arg237_1
    buf407 = reinterpret_tensor(buf406, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf406  # reuse
    buf408 = buf390; del buf390  # reuse
    buf409 = buf389; del buf389  # reuse
    buf411 = reinterpret_tensor(buf405, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf405  # reuse
    cpp_fused_add_native_layer_norm_76(c_void_p(buf407.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf411.data_ptr()))
    del arg238_1
    del arg239_1
    buf412 = buf388; del buf388  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg241_1, reinterpret_tensor(buf411, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg240_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf412)
    del arg240_1
    del arg241_1
    buf413 = reinterpret_tensor(buf411, (1024, 1024), (1024, 1), 0); del buf411  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg243_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg242_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf413)
    del arg242_1
    del arg243_1
    buf414 = buf381; del buf381  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg245_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg244_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf414)
    del arg244_1
    del arg245_1
    buf415 = reinterpret_tensor(buf360, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf360  # reuse
    buf416 = reinterpret_tensor(buf342, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf342  # reuse
    buf417 = buf395; del buf395  # reuse
    cpp_fused_clone_77(c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf418 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf415, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf416, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf417, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf419 = buf418[0]
    del buf418
    buf426 = reinterpret_tensor(buf419, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf419  # reuse
    cpp_fused_clone_78(c_void_p(buf426.data_ptr()))
    buf427 = reinterpret_tensor(buf417, (1024, 1024), (1024, 1), 0); del buf417  # reuse
    # Source Nodes: [hidden_states_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg247_1, reinterpret_tensor(buf426, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg246_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf427)
    del arg246_1
    del arg247_1
    buf428 = buf409; del buf409  # reuse
    buf429 = buf408; del buf408  # reuse
    buf431 = reinterpret_tensor(buf426, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf426  # reuse
    cpp_fused_add_native_layer_norm_79(c_void_p(buf407.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf431.data_ptr()))
    del arg248_1
    del arg249_1
    buf432 = reinterpret_tensor(buf387, (1024, 4096), (4096, 1), 0); del buf387  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg251_1, reinterpret_tensor(buf431, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg250_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf432)
    del arg250_1
    del arg251_1
    buf433 = reinterpret_tensor(buf432, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf432  # reuse
    cpp_fused_gelu_80(c_void_p(buf433.data_ptr()))
    buf434 = reinterpret_tensor(buf431, (1024, 1024), (1024, 1), 0); del buf431  # reuse
    # Source Nodes: [hidden_states_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg253_1, reinterpret_tensor(buf433, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg252_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf434)
    del arg252_1
    del arg253_1
    buf435 = buf429; del buf429  # reuse
    buf436 = buf428; del buf428  # reuse
    buf438 = reinterpret_tensor(buf416, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf416  # reuse
    cpp_fused_add_native_layer_norm_81(c_void_p(buf407.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf438.data_ptr()))
    del arg254_1
    del arg255_1
    buf439 = reinterpret_tensor(buf415, (1024, 1024), (1024, 1), 0); del buf415  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg257_1, reinterpret_tensor(buf438, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg256_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf439)
    del arg256_1
    del arg257_1
    buf440 = buf414; del buf414  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg259_1, reinterpret_tensor(buf438, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg258_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf440)
    del arg258_1
    del arg259_1
    buf441 = reinterpret_tensor(buf413, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf413  # reuse
    buf442 = reinterpret_tensor(buf412, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf412  # reuse
    cpp_fused_clone_82(c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()))
    buf443 = buf402; del buf402  # reuse
    # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf441, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf442, (16, 64, 1024), (65536, 1, 64), 0), out=buf443)
    buf444 = buf400; del buf400  # reuse
    buf445 = buf443; del buf443  # reuse
    buf446 = buf398; del buf398  # reuse
    cpp_fused__softmax_83(c_void_p(buf445.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf446.data_ptr()))
    buf447 = reinterpret_tensor(buf442, (1024, 1024), (1024, 1), 0); del buf442  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg261_1, reinterpret_tensor(buf438, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg260_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf447)
    del arg260_1
    del arg261_1
    buf448 = buf445; del buf445  # reuse
    buf449 = reinterpret_tensor(buf438, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf438  # reuse
    cpp_fused__softmax_clone_84(c_void_p(buf448.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf449.data_ptr()))
    buf450 = reinterpret_tensor(buf447, (16, 1024, 64), (65536, 64, 1), 0); del buf447  # reuse
    # Source Nodes: [attn_output_80, attn_weights_39], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf448, reinterpret_tensor(buf449, (16, 1024, 64), (65536, 64, 1), 0), out=buf450)
    buf451 = reinterpret_tensor(buf449, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf449  # reuse
    cpp_fused_clone_85(c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()))
    buf452 = reinterpret_tensor(buf450, (1024, 1024), (1024, 1), 0); del buf450  # reuse
    # Source Nodes: [hidden_states_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg263_1, reinterpret_tensor(buf451, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg262_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf452)
    del arg262_1
    del arg263_1
    buf453 = buf436; del buf436  # reuse
    buf454 = buf435; del buf435  # reuse
    buf456 = reinterpret_tensor(buf451, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf451  # reuse
    cpp_fused_add_native_layer_norm_86(c_void_p(buf407.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf456.data_ptr()))
    del arg264_1
    del arg265_1
    buf457 = reinterpret_tensor(buf441, (1024, 1024), (1024, 1), 0); del buf441  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg267_1, reinterpret_tensor(buf456, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg266_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf457)
    del arg266_1
    del arg267_1
    buf458 = reinterpret_tensor(buf456, (1024, 1024), (1024, 1), 0); del buf456  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg269_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg268_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf458)
    del arg268_1
    del arg269_1
    buf459 = buf440; del buf440  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg271_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg270_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf459)
    del arg270_1
    del arg271_1
    buf460 = reinterpret_tensor(buf439, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf439  # reuse
    buf461 = reinterpret_tensor(buf394, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf394  # reuse
    buf462 = reinterpret_tensor(buf393, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf393  # reuse
    cpp_fused_clone_87(c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()))
    del buf457
    del buf458
    # Source Nodes: [], Original ATen: []
    buf463 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf460, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf461, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf462, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf464 = buf463[0]
    del buf463
    buf471 = reinterpret_tensor(buf464, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf464  # reuse
    cpp_fused_clone_88(c_void_p(buf471.data_ptr()))
    buf472 = reinterpret_tensor(buf462, (1024, 1024), (1024, 1), 0); del buf462  # reuse
    # Source Nodes: [hidden_states_174], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg273_1, reinterpret_tensor(buf471, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg272_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf472)
    del arg272_1
    del arg273_1
    buf473 = reinterpret_tensor(buf472, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf472  # reuse
    buf474 = buf454; del buf454  # reuse
    buf475 = buf453; del buf453  # reuse
    buf477 = reinterpret_tensor(buf471, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf471  # reuse
    cpp_fused_add_native_layer_norm_89(c_void_p(buf473.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf477.data_ptr()))
    del arg274_1
    del arg275_1
    buf478 = reinterpret_tensor(buf433, (1024, 4096), (4096, 1), 0); del buf433  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg277_1, reinterpret_tensor(buf477, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg276_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf478)
    del arg276_1
    del arg277_1
    buf479 = reinterpret_tensor(buf478, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf478  # reuse
    cpp_fused_gelu_90(c_void_p(buf479.data_ptr()))
    buf480 = reinterpret_tensor(buf477, (1024, 1024), (1024, 1), 0); del buf477  # reuse
    # Source Nodes: [hidden_states_180], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg279_1, reinterpret_tensor(buf479, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg278_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf480)
    del arg278_1
    del arg279_1
    buf481 = buf475; del buf475  # reuse
    buf482 = buf474; del buf474  # reuse
    buf484 = reinterpret_tensor(buf452, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf452  # reuse
    cpp_fused_add_native_layer_norm_91(c_void_p(buf473.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf484.data_ptr()))
    del arg280_1
    del arg281_1
    buf485 = buf434; del buf434  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg283_1, reinterpret_tensor(buf484, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg282_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf485)
    del arg282_1
    del arg283_1
    buf486 = buf427; del buf427  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg285_1, reinterpret_tensor(buf484, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg284_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf486)
    del arg284_1
    del arg285_1
    buf487 = reinterpret_tensor(buf407, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf407  # reuse
    buf488 = buf461; del buf461  # reuse
    cpp_fused_clone_92(c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()))
    buf489 = buf448; del buf448  # reuse
    # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf487, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf488, (16, 64, 1024), (65536, 1, 64), 0), out=buf489)
    buf490 = buf446; del buf446  # reuse
    buf491 = buf489; del buf489  # reuse
    buf492 = buf444; del buf444  # reuse
    cpp_fused__softmax_93(c_void_p(buf491.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf492.data_ptr()))
    buf493 = reinterpret_tensor(buf488, (1024, 1024), (1024, 1), 0); del buf488  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg287_1, reinterpret_tensor(buf484, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg286_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf493)
    del arg286_1
    del arg287_1
    buf494 = buf491; del buf491  # reuse
    buf495 = reinterpret_tensor(buf484, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf484  # reuse
    cpp_fused__softmax_clone_94(c_void_p(buf494.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf495.data_ptr()))
    buf496 = reinterpret_tensor(buf493, (16, 1024, 64), (65536, 64, 1), 0); del buf493  # reuse
    # Source Nodes: [attn_output_90, attn_weights_45], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf494, reinterpret_tensor(buf495, (16, 1024, 64), (65536, 64, 1), 0), out=buf496)
    buf497 = reinterpret_tensor(buf495, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf495  # reuse
    cpp_fused_clone_95(c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()))
    buf498 = reinterpret_tensor(buf496, (1024, 1024), (1024, 1), 0); del buf496  # reuse
    # Source Nodes: [hidden_states_185], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg289_1, reinterpret_tensor(buf497, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg288_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf498)
    del arg288_1
    del arg289_1
    buf499 = buf482; del buf482  # reuse
    buf500 = buf481; del buf481  # reuse
    buf502 = reinterpret_tensor(buf497, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf497  # reuse
    cpp_fused_add_native_layer_norm_96(c_void_p(buf473.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf502.data_ptr()))
    del arg290_1
    del arg291_1
    buf503 = reinterpret_tensor(buf487, (1024, 1024), (1024, 1), 0); del buf487  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg293_1, reinterpret_tensor(buf502, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg292_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf503)
    del arg292_1
    del arg293_1
    buf504 = reinterpret_tensor(buf502, (1024, 1024), (1024, 1), 0); del buf502  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg295_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg294_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf504)
    del arg294_1
    del arg295_1
    buf505 = buf486; del buf486  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg297_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg296_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf505)
    del arg296_1
    del arg297_1
    buf506 = reinterpret_tensor(buf485, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf485  # reuse
    buf507 = buf460; del buf460  # reuse
    buf508 = reinterpret_tensor(buf459, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf459  # reuse
    cpp_fused_clone_97(c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()))
    del buf503
    del buf504
    # Source Nodes: [], Original ATen: []
    buf509 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf506, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf507, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf508, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf510 = buf509[0]
    del buf509
    buf517 = reinterpret_tensor(buf510, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf510  # reuse
    cpp_fused_clone_98(c_void_p(buf517.data_ptr()))
    buf518 = reinterpret_tensor(buf508, (1024, 1024), (1024, 1), 0); del buf508  # reuse
    # Source Nodes: [hidden_states_189], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg299_1, reinterpret_tensor(buf517, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg298_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf518)
    del arg298_1
    del arg299_1
    buf519 = buf500; del buf500  # reuse
    buf520 = buf499; del buf499  # reuse
    buf522 = reinterpret_tensor(buf517, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf517  # reuse
    cpp_fused_add_native_layer_norm_99(c_void_p(buf473.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf522.data_ptr()))
    del arg300_1
    del arg301_1
    buf523 = reinterpret_tensor(buf479, (1024, 4096), (4096, 1), 0); del buf479  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg303_1, reinterpret_tensor(buf522, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg302_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf523)
    del arg302_1
    del arg303_1
    buf524 = reinterpret_tensor(buf523, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf523  # reuse
    cpp_fused_gelu_100(c_void_p(buf524.data_ptr()))
    buf525 = reinterpret_tensor(buf522, (1024, 1024), (1024, 1), 0); del buf522  # reuse
    # Source Nodes: [hidden_states_195], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg305_1, reinterpret_tensor(buf524, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg304_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf525)
    del arg304_1
    del arg305_1
    buf526 = reinterpret_tensor(buf525, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf525  # reuse
    buf527 = buf520; del buf520  # reuse
    buf528 = buf519; del buf519  # reuse
    buf530 = reinterpret_tensor(buf507, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf507  # reuse
    cpp_fused_add_native_layer_norm_101(c_void_p(buf526.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf530.data_ptr()))
    del arg306_1
    del arg307_1
    buf531 = buf518; del buf518  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg309_1, reinterpret_tensor(buf530, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg308_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf531)
    del arg308_1
    del arg309_1
    buf532 = buf498; del buf498  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg311_1, reinterpret_tensor(buf530, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg310_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf532)
    del arg310_1
    del arg311_1
    buf533 = reinterpret_tensor(buf480, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf480  # reuse
    buf534 = reinterpret_tensor(buf473, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf473  # reuse
    cpp_fused_clone_102(c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()))
    buf535 = buf494; del buf494  # reuse
    # Source Nodes: [attn_weights_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf533, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf534, (16, 64, 1024), (65536, 1, 64), 0), out=buf535)
    buf536 = buf492; del buf492  # reuse
    buf537 = buf535; del buf535  # reuse
    buf538 = buf490; del buf490  # reuse
    cpp_fused__softmax_103(c_void_p(buf537.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf538.data_ptr()))
    buf539 = reinterpret_tensor(buf534, (1024, 1024), (1024, 1), 0); del buf534  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg313_1, reinterpret_tensor(buf530, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg312_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf539)
    del arg312_1
    del arg313_1
    buf540 = buf537; del buf537  # reuse
    buf541 = reinterpret_tensor(buf530, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf530  # reuse
    cpp_fused__softmax_clone_104(c_void_p(buf540.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf541.data_ptr()))
    buf542 = reinterpret_tensor(buf539, (16, 1024, 64), (65536, 64, 1), 0); del buf539  # reuse
    # Source Nodes: [attn_output_100, attn_weights_51], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf540, reinterpret_tensor(buf541, (16, 1024, 64), (65536, 64, 1), 0), out=buf542)
    buf543 = reinterpret_tensor(buf541, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf541  # reuse
    cpp_fused_clone_105(c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()))
    buf544 = reinterpret_tensor(buf542, (1024, 1024), (1024, 1), 0); del buf542  # reuse
    # Source Nodes: [hidden_states_200], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg315_1, reinterpret_tensor(buf543, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg314_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf544)
    del arg314_1
    del arg315_1
    buf545 = buf528; del buf528  # reuse
    buf546 = buf527; del buf527  # reuse
    buf548 = reinterpret_tensor(buf543, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf543  # reuse
    cpp_fused_add_native_layer_norm_106(c_void_p(buf526.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf548.data_ptr()))
    del arg316_1
    del arg317_1
    buf549 = reinterpret_tensor(buf533, (1024, 1024), (1024, 1), 0); del buf533  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg319_1, reinterpret_tensor(buf548, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg318_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf549)
    del arg318_1
    del arg319_1
    buf550 = reinterpret_tensor(buf548, (1024, 1024), (1024, 1), 0); del buf548  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg321_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg320_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf550)
    del arg320_1
    del arg321_1
    buf551 = buf532; del buf532  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg323_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg322_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf551)
    del arg322_1
    del arg323_1
    buf552 = reinterpret_tensor(buf531, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf531  # reuse
    buf553 = buf506; del buf506  # reuse
    buf554 = reinterpret_tensor(buf505, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf505  # reuse
    cpp_fused_clone_107(c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf555 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf552, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf553, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf554, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf556 = buf555[0]
    del buf555
    buf563 = reinterpret_tensor(buf556, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf556  # reuse
    cpp_fused_clone_108(c_void_p(buf563.data_ptr()))
    buf564 = reinterpret_tensor(buf554, (1024, 1024), (1024, 1), 0); del buf554  # reuse
    # Source Nodes: [hidden_states_204], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg325_1, reinterpret_tensor(buf563, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg324_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf564)
    del arg324_1
    del arg325_1
    buf565 = buf546; del buf546  # reuse
    buf566 = buf545; del buf545  # reuse
    buf568 = reinterpret_tensor(buf563, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf563  # reuse
    cpp_fused_add_native_layer_norm_109(c_void_p(buf526.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf568.data_ptr()))
    del arg326_1
    del arg327_1
    buf569 = reinterpret_tensor(buf524, (1024, 4096), (4096, 1), 0); del buf524  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg329_1, reinterpret_tensor(buf568, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg328_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf569)
    del arg328_1
    del arg329_1
    buf570 = reinterpret_tensor(buf569, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf569  # reuse
    cpp_fused_gelu_110(c_void_p(buf570.data_ptr()))
    buf571 = reinterpret_tensor(buf568, (1024, 1024), (1024, 1), 0); del buf568  # reuse
    # Source Nodes: [hidden_states_210], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg331_1, reinterpret_tensor(buf570, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg330_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf571)
    del arg330_1
    del arg331_1
    buf572 = buf566; del buf566  # reuse
    buf573 = buf565; del buf565  # reuse
    buf575 = reinterpret_tensor(buf553, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf553  # reuse
    cpp_fused_add_native_layer_norm_111(c_void_p(buf526.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf575.data_ptr()))
    del arg332_1
    del arg333_1
    buf576 = reinterpret_tensor(buf552, (1024, 1024), (1024, 1), 0); del buf552  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg335_1, reinterpret_tensor(buf575, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg334_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf576)
    del arg334_1
    del arg335_1
    buf577 = buf551; del buf551  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg337_1, reinterpret_tensor(buf575, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg336_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf577)
    del arg336_1
    del arg337_1
    buf578 = reinterpret_tensor(buf550, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf550  # reuse
    buf579 = reinterpret_tensor(buf549, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf549  # reuse
    cpp_fused_clone_112(c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()))
    buf580 = buf540; del buf540  # reuse
    # Source Nodes: [attn_weights_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf578, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf579, (16, 64, 1024), (65536, 1, 64), 0), out=buf580)
    buf581 = buf538; del buf538  # reuse
    buf582 = buf580; del buf580  # reuse
    buf583 = buf536; del buf536  # reuse
    cpp_fused__softmax_113(c_void_p(buf582.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf583.data_ptr()))
    buf584 = reinterpret_tensor(buf579, (1024, 1024), (1024, 1), 0); del buf579  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg339_1, reinterpret_tensor(buf575, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg338_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf584)
    del arg338_1
    del arg339_1
    buf585 = buf582; del buf582  # reuse
    buf586 = reinterpret_tensor(buf575, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf575  # reuse
    cpp_fused__softmax_clone_114(c_void_p(buf585.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf586.data_ptr()))
    buf587 = reinterpret_tensor(buf584, (16, 1024, 64), (65536, 64, 1), 0); del buf584  # reuse
    # Source Nodes: [attn_output_110, attn_weights_57], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf585, reinterpret_tensor(buf586, (16, 1024, 64), (65536, 64, 1), 0), out=buf587)
    buf588 = reinterpret_tensor(buf586, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf586  # reuse
    cpp_fused_clone_115(c_void_p(buf587.data_ptr()), c_void_p(buf588.data_ptr()))
    buf589 = reinterpret_tensor(buf587, (1024, 1024), (1024, 1), 0); del buf587  # reuse
    # Source Nodes: [hidden_states_215], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg341_1, reinterpret_tensor(buf588, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg340_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf589)
    del arg340_1
    del arg341_1
    buf590 = reinterpret_tensor(buf589, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf589  # reuse
    buf591 = buf573; del buf573  # reuse
    buf592 = buf572; del buf572  # reuse
    buf594 = reinterpret_tensor(buf588, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf588  # reuse
    cpp_fused_add_native_layer_norm_116(c_void_p(buf590.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf594.data_ptr()))
    del arg342_1
    del arg343_1
    buf595 = buf571; del buf571  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg345_1, reinterpret_tensor(buf594, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg344_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf595)
    del arg344_1
    del arg345_1
    buf596 = reinterpret_tensor(buf594, (1024, 1024), (1024, 1), 0); del buf594  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg347_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg346_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf596)
    del arg346_1
    del arg347_1
    buf597 = buf564; del buf564  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg349_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg348_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf597)
    del arg348_1
    del arg349_1
    buf598 = reinterpret_tensor(buf544, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf544  # reuse
    buf599 = reinterpret_tensor(buf526, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf526  # reuse
    buf600 = buf578; del buf578  # reuse
    cpp_fused_clone_117(c_void_p(buf595.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf600.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf601 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf598, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf599, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf600, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf602 = buf601[0]
    del buf601
    buf609 = reinterpret_tensor(buf602, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf602  # reuse
    cpp_fused_clone_118(c_void_p(buf609.data_ptr()))
    buf610 = reinterpret_tensor(buf600, (1024, 1024), (1024, 1), 0); del buf600  # reuse
    # Source Nodes: [hidden_states_219], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg351_1, reinterpret_tensor(buf609, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg350_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf610)
    del arg350_1
    del arg351_1
    buf611 = buf592; del buf592  # reuse
    buf612 = buf591; del buf591  # reuse
    buf614 = reinterpret_tensor(buf609, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf609  # reuse
    cpp_fused_add_native_layer_norm_119(c_void_p(buf590.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf614.data_ptr()))
    del arg352_1
    del arg353_1
    buf615 = reinterpret_tensor(buf570, (1024, 4096), (4096, 1), 0); del buf570  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg355_1, reinterpret_tensor(buf614, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg354_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf615)
    del arg354_1
    del arg355_1
    buf616 = reinterpret_tensor(buf615, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf615  # reuse
    cpp_fused_gelu_120(c_void_p(buf616.data_ptr()))
    buf617 = reinterpret_tensor(buf614, (1024, 1024), (1024, 1), 0); del buf614  # reuse
    # Source Nodes: [hidden_states_225], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg357_1, reinterpret_tensor(buf616, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg356_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf617)
    del arg356_1
    del arg357_1
    buf618 = buf612; del buf612  # reuse
    buf619 = buf611; del buf611  # reuse
    buf621 = reinterpret_tensor(buf599, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf599  # reuse
    cpp_fused_add_native_layer_norm_121(c_void_p(buf590.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf621.data_ptr()))
    del arg358_1
    del arg359_1
    buf622 = reinterpret_tensor(buf598, (1024, 1024), (1024, 1), 0); del buf598  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg361_1, reinterpret_tensor(buf621, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg360_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf622)
    del arg360_1
    del arg361_1
    buf623 = buf597; del buf597  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg363_1, reinterpret_tensor(buf621, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg362_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf623)
    del arg362_1
    del arg363_1
    buf624 = reinterpret_tensor(buf596, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf596  # reuse
    buf625 = reinterpret_tensor(buf595, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf595  # reuse
    cpp_fused_clone_122(c_void_p(buf622.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf625.data_ptr()))
    buf626 = buf585; del buf585  # reuse
    # Source Nodes: [attn_weights_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf624, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf625, (16, 64, 1024), (65536, 1, 64), 0), out=buf626)
    buf627 = buf583; del buf583  # reuse
    buf628 = buf626; del buf626  # reuse
    buf629 = buf581; del buf581  # reuse
    cpp_fused__softmax_123(c_void_p(buf628.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf629.data_ptr()))
    buf630 = reinterpret_tensor(buf625, (1024, 1024), (1024, 1), 0); del buf625  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg365_1, reinterpret_tensor(buf621, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg364_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf630)
    del arg364_1
    del arg365_1
    buf631 = buf628; del buf628  # reuse
    buf632 = reinterpret_tensor(buf621, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf621  # reuse
    cpp_fused__softmax_clone_124(c_void_p(buf631.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf632.data_ptr()))
    buf633 = reinterpret_tensor(buf630, (16, 1024, 64), (65536, 64, 1), 0); del buf630  # reuse
    # Source Nodes: [attn_output_120, attn_weights_63], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf631, reinterpret_tensor(buf632, (16, 1024, 64), (65536, 64, 1), 0), out=buf633)
    buf634 = reinterpret_tensor(buf632, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf632  # reuse
    cpp_fused_clone_125(c_void_p(buf633.data_ptr()), c_void_p(buf634.data_ptr()))
    buf635 = reinterpret_tensor(buf633, (1024, 1024), (1024, 1), 0); del buf633  # reuse
    # Source Nodes: [hidden_states_230], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg367_1, reinterpret_tensor(buf634, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg366_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf635)
    del arg366_1
    del arg367_1
    buf636 = buf619; del buf619  # reuse
    buf637 = buf618; del buf618  # reuse
    buf639 = reinterpret_tensor(buf634, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf634  # reuse
    cpp_fused_add_native_layer_norm_126(c_void_p(buf590.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(buf636.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf639.data_ptr()))
    del arg368_1
    del arg369_1
    buf640 = reinterpret_tensor(buf624, (1024, 1024), (1024, 1), 0); del buf624  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg371_1, reinterpret_tensor(buf639, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg370_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf640)
    del arg370_1
    del arg371_1
    buf641 = reinterpret_tensor(buf639, (1024, 1024), (1024, 1), 0); del buf639  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg373_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg372_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf641)
    del arg372_1
    del arg373_1
    buf642 = buf623; del buf623  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg375_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg374_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf642)
    del arg374_1
    del arg375_1
    buf643 = reinterpret_tensor(buf622, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf622  # reuse
    buf644 = reinterpret_tensor(buf577, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf577  # reuse
    buf645 = reinterpret_tensor(buf576, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf576  # reuse
    cpp_fused_clone_127(c_void_p(buf640.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf643.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf645.data_ptr()))
    del buf640
    del buf641
    # Source Nodes: [], Original ATen: []
    buf646 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf643, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf644, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf645, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf647 = buf646[0]
    del buf646
    buf654 = reinterpret_tensor(buf647, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf647  # reuse
    cpp_fused_clone_128(c_void_p(buf654.data_ptr()))
    buf655 = reinterpret_tensor(buf645, (1024, 1024), (1024, 1), 0); del buf645  # reuse
    # Source Nodes: [hidden_states_234], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg377_1, reinterpret_tensor(buf654, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg376_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf655)
    del arg376_1
    del arg377_1
    buf656 = reinterpret_tensor(buf655, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf655  # reuse
    buf657 = buf637; del buf637  # reuse
    buf658 = buf636; del buf636  # reuse
    buf660 = reinterpret_tensor(buf654, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf654  # reuse
    cpp_fused_add_native_layer_norm_129(c_void_p(buf656.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(arg378_1.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf660.data_ptr()))
    del arg378_1
    del arg379_1
    buf661 = reinterpret_tensor(buf616, (1024, 4096), (4096, 1), 0); del buf616  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg381_1, reinterpret_tensor(buf660, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg380_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf661)
    del arg380_1
    del arg381_1
    buf662 = reinterpret_tensor(buf661, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf661  # reuse
    cpp_fused_gelu_130(c_void_p(buf662.data_ptr()))
    buf663 = reinterpret_tensor(buf660, (1024, 1024), (1024, 1), 0); del buf660  # reuse
    # Source Nodes: [hidden_states_240], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg383_1, reinterpret_tensor(buf662, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg382_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf663)
    del arg382_1
    del arg383_1
    buf664 = buf658; del buf658  # reuse
    buf665 = buf657; del buf657  # reuse
    buf667 = reinterpret_tensor(buf635, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf635  # reuse
    cpp_fused_add_native_layer_norm_131(c_void_p(buf656.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf667.data_ptr()))
    del arg384_1
    del arg385_1
    buf668 = buf617; del buf617  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg387_1, reinterpret_tensor(buf667, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg386_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf668)
    del arg386_1
    del arg387_1
    buf669 = buf610; del buf610  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg389_1, reinterpret_tensor(buf667, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg388_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf669)
    del arg388_1
    del arg389_1
    buf670 = reinterpret_tensor(buf590, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf590  # reuse
    buf671 = buf644; del buf644  # reuse
    cpp_fused_clone_132(c_void_p(buf668.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf670.data_ptr()), c_void_p(buf671.data_ptr()))
    buf672 = buf631; del buf631  # reuse
    # Source Nodes: [attn_weights_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf670, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf671, (16, 64, 1024), (65536, 1, 64), 0), out=buf672)
    buf673 = buf629; del buf629  # reuse
    buf674 = buf672; del buf672  # reuse
    buf675 = buf627; del buf627  # reuse
    cpp_fused__softmax_133(c_void_p(buf674.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf675.data_ptr()))
    buf676 = reinterpret_tensor(buf671, (1024, 1024), (1024, 1), 0); del buf671  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg391_1, reinterpret_tensor(buf667, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg390_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf676)
    del arg390_1
    del arg391_1
    buf677 = buf674; del buf674  # reuse
    buf678 = reinterpret_tensor(buf667, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf667  # reuse
    cpp_fused__softmax_clone_134(c_void_p(buf677.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf676.data_ptr()), c_void_p(buf678.data_ptr()))
    buf679 = reinterpret_tensor(buf676, (16, 1024, 64), (65536, 64, 1), 0); del buf676  # reuse
    # Source Nodes: [attn_output_130, attn_weights_69], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf677, reinterpret_tensor(buf678, (16, 1024, 64), (65536, 64, 1), 0), out=buf679)
    buf680 = reinterpret_tensor(buf678, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf678  # reuse
    cpp_fused_clone_135(c_void_p(buf679.data_ptr()), c_void_p(buf680.data_ptr()))
    buf681 = reinterpret_tensor(buf679, (1024, 1024), (1024, 1), 0); del buf679  # reuse
    # Source Nodes: [hidden_states_245], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg393_1, reinterpret_tensor(buf680, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg392_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf681)
    del arg392_1
    del arg393_1
    buf682 = buf665; del buf665  # reuse
    buf683 = buf664; del buf664  # reuse
    buf685 = reinterpret_tensor(buf680, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf680  # reuse
    cpp_fused_add_native_layer_norm_136(c_void_p(buf656.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(arg394_1.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf685.data_ptr()))
    del arg394_1
    del arg395_1
    buf686 = reinterpret_tensor(buf670, (1024, 1024), (1024, 1), 0); del buf670  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg397_1, reinterpret_tensor(buf685, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg396_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf686)
    del arg396_1
    del arg397_1
    buf687 = reinterpret_tensor(buf685, (1024, 1024), (1024, 1), 0); del buf685  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg399_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg398_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf687)
    del arg398_1
    del arg399_1
    buf688 = buf669; del buf669  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg401_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg400_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf688)
    del arg400_1
    del arg401_1
    buf689 = reinterpret_tensor(buf668, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf668  # reuse
    buf690 = buf643; del buf643  # reuse
    buf691 = reinterpret_tensor(buf642, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf642  # reuse
    cpp_fused_clone_137(c_void_p(buf686.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf691.data_ptr()))
    del buf686
    del buf687
    # Source Nodes: [], Original ATen: []
    buf692 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf689, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf690, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf691, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf693 = buf692[0]
    del buf692
    buf700 = reinterpret_tensor(buf693, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf693  # reuse
    cpp_fused_clone_138(c_void_p(buf700.data_ptr()))
    buf701 = reinterpret_tensor(buf691, (1024, 1024), (1024, 1), 0); del buf691  # reuse
    # Source Nodes: [hidden_states_249], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg403_1, reinterpret_tensor(buf700, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg402_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf701)
    del arg402_1
    del arg403_1
    buf702 = buf683; del buf683  # reuse
    buf703 = buf682; del buf682  # reuse
    buf705 = reinterpret_tensor(buf700, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf700  # reuse
    cpp_fused_add_native_layer_norm_139(c_void_p(buf656.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(arg404_1.data_ptr()), c_void_p(arg405_1.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf703.data_ptr()), c_void_p(buf705.data_ptr()))
    del arg404_1
    del arg405_1
    buf706 = reinterpret_tensor(buf662, (1024, 4096), (4096, 1), 0); del buf662  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg407_1, reinterpret_tensor(buf705, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg406_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf706)
    del arg406_1
    del arg407_1
    buf707 = reinterpret_tensor(buf706, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf706  # reuse
    cpp_fused_gelu_140(c_void_p(buf707.data_ptr()))
    buf708 = reinterpret_tensor(buf705, (1024, 1024), (1024, 1), 0); del buf705  # reuse
    # Source Nodes: [hidden_states_255], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg409_1, reinterpret_tensor(buf707, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg408_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf708)
    del arg408_1
    del arg409_1
    buf709 = reinterpret_tensor(buf708, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf708  # reuse
    buf710 = buf703; del buf703  # reuse
    buf711 = buf702; del buf702  # reuse
    buf713 = reinterpret_tensor(buf690, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf690  # reuse
    cpp_fused_add_native_layer_norm_141(c_void_p(buf709.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(arg410_1.data_ptr()), c_void_p(arg411_1.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf713.data_ptr()))
    del arg410_1
    del arg411_1
    buf714 = buf701; del buf701  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg413_1, reinterpret_tensor(buf713, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg412_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf714)
    del arg412_1
    del arg413_1
    buf715 = buf681; del buf681  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg415_1, reinterpret_tensor(buf713, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg414_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf715)
    del arg414_1
    del arg415_1
    buf716 = reinterpret_tensor(buf663, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf663  # reuse
    buf717 = reinterpret_tensor(buf656, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf656  # reuse
    cpp_fused_clone_142(c_void_p(buf714.data_ptr()), c_void_p(buf715.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf717.data_ptr()))
    buf718 = buf677; del buf677  # reuse
    # Source Nodes: [attn_weights_72], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf716, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf717, (16, 64, 1024), (65536, 1, 64), 0), out=buf718)
    buf719 = buf675; del buf675  # reuse
    buf720 = buf718; del buf718  # reuse
    buf721 = buf673; del buf673  # reuse
    cpp_fused__softmax_143(c_void_p(buf720.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf721.data_ptr()))
    buf722 = reinterpret_tensor(buf717, (1024, 1024), (1024, 1), 0); del buf717  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg417_1, reinterpret_tensor(buf713, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg416_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf722)
    del arg416_1
    del arg417_1
    buf723 = buf720; del buf720  # reuse
    buf724 = reinterpret_tensor(buf713, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf713  # reuse
    cpp_fused__softmax_clone_144(c_void_p(buf723.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf722.data_ptr()), c_void_p(buf724.data_ptr()))
    buf725 = reinterpret_tensor(buf722, (16, 1024, 64), (65536, 64, 1), 0); del buf722  # reuse
    # Source Nodes: [attn_output_140, attn_weights_75], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf723, reinterpret_tensor(buf724, (16, 1024, 64), (65536, 64, 1), 0), out=buf725)
    buf726 = reinterpret_tensor(buf724, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf724  # reuse
    cpp_fused_clone_145(c_void_p(buf725.data_ptr()), c_void_p(buf726.data_ptr()))
    buf727 = reinterpret_tensor(buf725, (1024, 1024), (1024, 1), 0); del buf725  # reuse
    # Source Nodes: [hidden_states_260], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg419_1, reinterpret_tensor(buf726, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg418_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf727)
    del arg418_1
    del arg419_1
    buf728 = buf711; del buf711  # reuse
    buf729 = buf710; del buf710  # reuse
    buf731 = reinterpret_tensor(buf726, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf726  # reuse
    cpp_fused_add_native_layer_norm_146(c_void_p(buf709.data_ptr()), c_void_p(buf727.data_ptr()), c_void_p(arg420_1.data_ptr()), c_void_p(arg421_1.data_ptr()), c_void_p(buf728.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf731.data_ptr()))
    del arg420_1
    del arg421_1
    buf732 = reinterpret_tensor(buf716, (1024, 1024), (1024, 1), 0); del buf716  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg423_1, reinterpret_tensor(buf731, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg422_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf732)
    del arg422_1
    del arg423_1
    buf733 = reinterpret_tensor(buf731, (1024, 1024), (1024, 1), 0); del buf731  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg425_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg424_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf733)
    del arg424_1
    del arg425_1
    buf734 = buf715; del buf715  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg427_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg426_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf734)
    del arg426_1
    del arg427_1
    buf735 = reinterpret_tensor(buf714, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf714  # reuse
    buf736 = buf689; del buf689  # reuse
    buf737 = reinterpret_tensor(buf688, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf688  # reuse
    cpp_fused_clone_147(c_void_p(buf732.data_ptr()), c_void_p(buf733.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(buf735.data_ptr()), c_void_p(buf736.data_ptr()), c_void_p(buf737.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf738 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf735, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf736, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf737, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf739 = buf738[0]
    del buf738
    buf746 = reinterpret_tensor(buf739, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf739  # reuse
    cpp_fused_clone_148(c_void_p(buf746.data_ptr()))
    buf747 = reinterpret_tensor(buf737, (1024, 1024), (1024, 1), 0); del buf737  # reuse
    # Source Nodes: [hidden_states_264], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg429_1, reinterpret_tensor(buf746, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg428_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf747)
    del arg428_1
    del arg429_1
    buf748 = buf729; del buf729  # reuse
    buf749 = buf728; del buf728  # reuse
    buf751 = reinterpret_tensor(buf746, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf746  # reuse
    cpp_fused_add_native_layer_norm_149(c_void_p(buf709.data_ptr()), c_void_p(buf727.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(arg430_1.data_ptr()), c_void_p(arg431_1.data_ptr()), c_void_p(buf748.data_ptr()), c_void_p(buf749.data_ptr()), c_void_p(buf751.data_ptr()))
    del arg430_1
    del arg431_1
    buf752 = reinterpret_tensor(buf707, (1024, 4096), (4096, 1), 0); del buf707  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg433_1, reinterpret_tensor(buf751, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg432_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf752)
    del arg432_1
    del arg433_1
    buf753 = reinterpret_tensor(buf752, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf752  # reuse
    cpp_fused_gelu_150(c_void_p(buf753.data_ptr()))
    buf754 = reinterpret_tensor(buf751, (1024, 1024), (1024, 1), 0); del buf751  # reuse
    # Source Nodes: [hidden_states_270], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg435_1, reinterpret_tensor(buf753, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg434_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf754)
    del arg434_1
    del arg435_1
    buf755 = buf749; del buf749  # reuse
    buf756 = buf748; del buf748  # reuse
    buf758 = reinterpret_tensor(buf736, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf736  # reuse
    cpp_fused_add_native_layer_norm_151(c_void_p(buf709.data_ptr()), c_void_p(buf727.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(arg436_1.data_ptr()), c_void_p(arg437_1.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf756.data_ptr()), c_void_p(buf758.data_ptr()))
    del arg436_1
    del arg437_1
    buf759 = reinterpret_tensor(buf735, (1024, 1024), (1024, 1), 0); del buf735  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg439_1, reinterpret_tensor(buf758, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg438_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf759)
    del arg438_1
    del arg439_1
    buf760 = buf734; del buf734  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg441_1, reinterpret_tensor(buf758, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg440_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf760)
    del arg440_1
    del arg441_1
    buf761 = reinterpret_tensor(buf733, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf733  # reuse
    buf762 = reinterpret_tensor(buf732, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf732  # reuse
    cpp_fused_clone_152(c_void_p(buf759.data_ptr()), c_void_p(buf760.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(buf762.data_ptr()))
    buf763 = buf723; del buf723  # reuse
    # Source Nodes: [attn_weights_78], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf761, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf762, (16, 64, 1024), (65536, 1, 64), 0), out=buf763)
    buf764 = buf721; del buf721  # reuse
    buf765 = buf763; del buf763  # reuse
    buf766 = buf719; del buf719  # reuse
    cpp_fused__softmax_153(c_void_p(buf765.data_ptr()), c_void_p(buf764.data_ptr()), c_void_p(buf766.data_ptr()))
    buf767 = reinterpret_tensor(buf762, (1024, 1024), (1024, 1), 0); del buf762  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg443_1, reinterpret_tensor(buf758, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg442_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf767)
    del arg442_1
    del arg443_1
    buf768 = buf765; del buf765  # reuse
    buf769 = reinterpret_tensor(buf758, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf758  # reuse
    cpp_fused__softmax_clone_154(c_void_p(buf768.data_ptr()), c_void_p(buf766.data_ptr()), c_void_p(buf767.data_ptr()), c_void_p(buf769.data_ptr()))
    buf770 = reinterpret_tensor(buf767, (16, 1024, 64), (65536, 64, 1), 0); del buf767  # reuse
    # Source Nodes: [attn_output_150, attn_weights_81], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf768, reinterpret_tensor(buf769, (16, 1024, 64), (65536, 64, 1), 0), out=buf770)
    buf771 = reinterpret_tensor(buf769, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf769  # reuse
    cpp_fused_clone_155(c_void_p(buf770.data_ptr()), c_void_p(buf771.data_ptr()))
    buf772 = reinterpret_tensor(buf770, (1024, 1024), (1024, 1), 0); del buf770  # reuse
    # Source Nodes: [hidden_states_275], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg445_1, reinterpret_tensor(buf771, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg444_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf772)
    del arg444_1
    del arg445_1
    buf773 = reinterpret_tensor(buf772, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf772  # reuse
    buf774 = buf756; del buf756  # reuse
    buf775 = buf755; del buf755  # reuse
    buf777 = reinterpret_tensor(buf771, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf771  # reuse
    cpp_fused_add_native_layer_norm_156(c_void_p(buf773.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(buf727.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(arg446_1.data_ptr()), c_void_p(arg447_1.data_ptr()), c_void_p(buf774.data_ptr()), c_void_p(buf775.data_ptr()), c_void_p(buf777.data_ptr()))
    del arg446_1
    del arg447_1
    buf778 = buf754; del buf754  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg449_1, reinterpret_tensor(buf777, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg448_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf778)
    del arg448_1
    del arg449_1
    buf779 = reinterpret_tensor(buf777, (1024, 1024), (1024, 1), 0); del buf777  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg451_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg450_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf779)
    del arg450_1
    del arg451_1
    buf780 = buf747; del buf747  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg453_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg452_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf780)
    del arg452_1
    del arg453_1
    buf781 = reinterpret_tensor(buf727, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf727  # reuse
    buf782 = reinterpret_tensor(buf709, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf709  # reuse
    buf783 = buf761; del buf761  # reuse
    cpp_fused_clone_157(c_void_p(buf778.data_ptr()), c_void_p(buf779.data_ptr()), c_void_p(buf780.data_ptr()), c_void_p(buf781.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(buf783.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf784 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf781, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf782, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf783, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf785 = buf784[0]
    del buf784
    buf792 = reinterpret_tensor(buf785, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf785  # reuse
    cpp_fused_clone_158(c_void_p(buf792.data_ptr()))
    buf793 = reinterpret_tensor(buf783, (1024, 1024), (1024, 1), 0); del buf783  # reuse
    # Source Nodes: [hidden_states_279], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg455_1, reinterpret_tensor(buf792, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg454_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf793)
    del arg454_1
    del arg455_1
    buf794 = buf775; del buf775  # reuse
    buf795 = buf774; del buf774  # reuse
    buf797 = reinterpret_tensor(buf792, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf792  # reuse
    cpp_fused_add_native_layer_norm_159(c_void_p(buf773.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(arg456_1.data_ptr()), c_void_p(arg457_1.data_ptr()), c_void_p(buf794.data_ptr()), c_void_p(buf795.data_ptr()), c_void_p(buf797.data_ptr()))
    del arg456_1
    del arg457_1
    buf798 = reinterpret_tensor(buf753, (1024, 4096), (4096, 1), 0); del buf753  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg459_1, reinterpret_tensor(buf797, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg458_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf798)
    del arg458_1
    del arg459_1
    buf799 = reinterpret_tensor(buf798, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf798  # reuse
    cpp_fused_gelu_160(c_void_p(buf799.data_ptr()))
    buf800 = reinterpret_tensor(buf797, (1024, 1024), (1024, 1), 0); del buf797  # reuse
    # Source Nodes: [hidden_states_285], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg461_1, reinterpret_tensor(buf799, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg460_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf800)
    del arg460_1
    del arg461_1
    buf801 = buf795; del buf795  # reuse
    buf802 = buf794; del buf794  # reuse
    buf804 = reinterpret_tensor(buf782, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf782  # reuse
    cpp_fused_add_native_layer_norm_161(c_void_p(buf773.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(arg462_1.data_ptr()), c_void_p(arg463_1.data_ptr()), c_void_p(buf801.data_ptr()), c_void_p(buf802.data_ptr()), c_void_p(buf804.data_ptr()))
    del arg462_1
    del arg463_1
    buf805 = reinterpret_tensor(buf781, (1024, 1024), (1024, 1), 0); del buf781  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg465_1, reinterpret_tensor(buf804, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg464_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf805)
    del arg464_1
    del arg465_1
    buf806 = buf780; del buf780  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg467_1, reinterpret_tensor(buf804, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg466_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf806)
    del arg466_1
    del arg467_1
    buf807 = reinterpret_tensor(buf779, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf779  # reuse
    buf808 = reinterpret_tensor(buf778, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf778  # reuse
    cpp_fused_clone_162(c_void_p(buf805.data_ptr()), c_void_p(buf806.data_ptr()), c_void_p(buf807.data_ptr()), c_void_p(buf808.data_ptr()))
    buf809 = buf768; del buf768  # reuse
    # Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf807, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf808, (16, 64, 1024), (65536, 1, 64), 0), out=buf809)
    buf810 = buf766; del buf766  # reuse
    buf811 = buf809; del buf809  # reuse
    buf812 = buf764; del buf764  # reuse
    cpp_fused__softmax_163(c_void_p(buf811.data_ptr()), c_void_p(buf810.data_ptr()), c_void_p(buf812.data_ptr()))
    buf813 = reinterpret_tensor(buf808, (1024, 1024), (1024, 1), 0); del buf808  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg469_1, reinterpret_tensor(buf804, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg468_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf813)
    del arg468_1
    del arg469_1
    buf814 = buf811; del buf811  # reuse
    buf815 = reinterpret_tensor(buf804, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf804  # reuse
    cpp_fused__softmax_clone_164(c_void_p(buf814.data_ptr()), c_void_p(buf812.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf815.data_ptr()))
    buf816 = reinterpret_tensor(buf813, (16, 1024, 64), (65536, 64, 1), 0); del buf813  # reuse
    # Source Nodes: [attn_output_160, attn_weights_87], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf814, reinterpret_tensor(buf815, (16, 1024, 64), (65536, 64, 1), 0), out=buf816)
    buf817 = reinterpret_tensor(buf815, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf815  # reuse
    cpp_fused_clone_165(c_void_p(buf816.data_ptr()), c_void_p(buf817.data_ptr()))
    buf818 = reinterpret_tensor(buf816, (1024, 1024), (1024, 1), 0); del buf816  # reuse
    # Source Nodes: [hidden_states_290], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg471_1, reinterpret_tensor(buf817, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg470_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf818)
    del arg470_1
    del arg471_1
    buf819 = buf802; del buf802  # reuse
    buf820 = buf801; del buf801  # reuse
    buf822 = reinterpret_tensor(buf817, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf817  # reuse
    cpp_fused_add_native_layer_norm_166(c_void_p(buf773.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf818.data_ptr()), c_void_p(arg472_1.data_ptr()), c_void_p(arg473_1.data_ptr()), c_void_p(buf819.data_ptr()), c_void_p(buf820.data_ptr()), c_void_p(buf822.data_ptr()))
    del arg472_1
    del arg473_1
    buf823 = reinterpret_tensor(buf807, (1024, 1024), (1024, 1), 0); del buf807  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg475_1, reinterpret_tensor(buf822, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg474_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf823)
    del arg474_1
    del arg475_1
    buf824 = reinterpret_tensor(buf822, (1024, 1024), (1024, 1), 0); del buf822  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg477_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg476_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf824)
    del arg476_1
    del arg477_1
    buf825 = buf806; del buf806  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg479_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg478_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf825)
    del arg478_1
    del arg479_1
    buf826 = reinterpret_tensor(buf805, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf805  # reuse
    buf827 = reinterpret_tensor(buf760, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf760  # reuse
    buf828 = reinterpret_tensor(buf759, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf759  # reuse
    cpp_fused_clone_167(c_void_p(buf823.data_ptr()), c_void_p(buf824.data_ptr()), c_void_p(buf825.data_ptr()), c_void_p(buf826.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf828.data_ptr()))
    del buf823
    del buf824
    # Source Nodes: [], Original ATen: []
    buf829 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf826, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf827, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf828, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    buf830 = buf829[0]
    del buf829
    buf837 = reinterpret_tensor(buf830, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf830  # reuse
    cpp_fused_clone_168(c_void_p(buf837.data_ptr()))
    buf838 = reinterpret_tensor(buf828, (1024, 1024), (1024, 1), 0); del buf828  # reuse
    # Source Nodes: [hidden_states_294], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg481_1, reinterpret_tensor(buf837, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg480_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf838)
    del arg480_1
    del arg481_1
    buf839 = reinterpret_tensor(buf838, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf838  # reuse
    buf840 = buf820; del buf820  # reuse
    buf841 = buf819; del buf819  # reuse
    buf843 = reinterpret_tensor(buf837, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf837  # reuse
    cpp_fused_add_native_layer_norm_169(c_void_p(buf839.data_ptr()), c_void_p(buf773.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf818.data_ptr()), c_void_p(arg482_1.data_ptr()), c_void_p(arg483_1.data_ptr()), c_void_p(buf840.data_ptr()), c_void_p(buf841.data_ptr()), c_void_p(buf843.data_ptr()))
    del arg482_1
    del arg483_1
    buf844 = reinterpret_tensor(buf799, (1024, 4096), (4096, 1), 0); del buf799  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg485_1, reinterpret_tensor(buf843, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg484_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf844)
    del arg484_1
    del arg485_1
    buf845 = reinterpret_tensor(buf844, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf844  # reuse
    cpp_fused_gelu_170(c_void_p(buf845.data_ptr()))
    buf846 = reinterpret_tensor(buf843, (1024, 1024), (1024, 1), 0); del buf843  # reuse
    # Source Nodes: [hidden_states_300], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg487_1, reinterpret_tensor(buf845, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg486_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf846)
    del arg486_1
    del arg487_1
    buf847 = buf841; del buf841  # reuse
    buf848 = buf840; del buf840  # reuse
    buf850 = reinterpret_tensor(buf818, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf818  # reuse
    cpp_fused_add_native_layer_norm_171(c_void_p(buf839.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(arg488_1.data_ptr()), c_void_p(arg489_1.data_ptr()), c_void_p(buf847.data_ptr()), c_void_p(buf848.data_ptr()), c_void_p(buf850.data_ptr()))
    del arg488_1
    del arg489_1
    buf851 = buf800; del buf800  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg491_1, reinterpret_tensor(buf850, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg490_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf851)
    del arg490_1
    del arg491_1
    buf852 = buf793; del buf793  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg493_1, reinterpret_tensor(buf850, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg492_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf852)
    del arg492_1
    del arg493_1
    buf853 = reinterpret_tensor(buf773, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf773  # reuse
    buf854 = buf827; del buf827  # reuse
    cpp_fused_clone_172(c_void_p(buf851.data_ptr()), c_void_p(buf852.data_ptr()), c_void_p(buf853.data_ptr()), c_void_p(buf854.data_ptr()))
    buf855 = buf814; del buf814  # reuse
    # Source Nodes: [attn_weights_90], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf853, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf854, (16, 64, 1024), (65536, 1, 64), 0), out=buf855)
    buf856 = buf812; del buf812  # reuse
    buf857 = buf855; del buf855  # reuse
    buf858 = buf810; del buf810  # reuse
    cpp_fused__softmax_173(c_void_p(buf857.data_ptr()), c_void_p(buf856.data_ptr()), c_void_p(buf858.data_ptr()))
    del buf856
    buf859 = reinterpret_tensor(buf854, (1024, 1024), (1024, 1), 0); del buf854  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg495_1, reinterpret_tensor(buf850, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg494_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf859)
    del arg494_1
    del arg495_1
    buf860 = buf857; del buf857  # reuse
    buf861 = reinterpret_tensor(buf850, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf850  # reuse
    cpp_fused__softmax_clone_174(c_void_p(buf860.data_ptr()), c_void_p(buf858.data_ptr()), c_void_p(buf859.data_ptr()), c_void_p(buf861.data_ptr()))
    del buf858
    buf862 = reinterpret_tensor(buf859, (16, 1024, 64), (65536, 64, 1), 0); del buf859  # reuse
    # Source Nodes: [attn_output_170, attn_weights_93], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf860, reinterpret_tensor(buf861, (16, 1024, 64), (65536, 64, 1), 0), out=buf862)
    del buf860
    buf863 = reinterpret_tensor(buf861, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf861  # reuse
    cpp_fused_clone_175(c_void_p(buf862.data_ptr()), c_void_p(buf863.data_ptr()))
    buf864 = reinterpret_tensor(buf862, (1024, 1024), (1024, 1), 0); del buf862  # reuse
    # Source Nodes: [hidden_states_305], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg497_1, reinterpret_tensor(buf863, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg496_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf864)
    del arg496_1
    del arg497_1
    buf865 = buf848; del buf848  # reuse
    buf866 = buf847; del buf847  # reuse
    buf868 = reinterpret_tensor(buf863, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf863  # reuse
    cpp_fused_add_native_layer_norm_176(c_void_p(buf839.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(buf864.data_ptr()), c_void_p(arg498_1.data_ptr()), c_void_p(arg499_1.data_ptr()), c_void_p(buf865.data_ptr()), c_void_p(buf866.data_ptr()), c_void_p(buf868.data_ptr()))
    del arg498_1
    del arg499_1
    buf869 = reinterpret_tensor(buf853, (1024, 1024), (1024, 1), 0); del buf853  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg501_1, reinterpret_tensor(buf868, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg500_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf869)
    del arg500_1
    del arg501_1
    buf870 = reinterpret_tensor(buf868, (1024, 1024), (1024, 1), 0); del buf868  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg503_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg502_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf870)
    del arg502_1
    del arg503_1
    buf871 = buf852; del buf852  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg505_1, reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg504_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf871)
    del arg504_1
    del arg505_1
    buf872 = reinterpret_tensor(buf851, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf851  # reuse
    buf873 = buf826; del buf826  # reuse
    buf874 = reinterpret_tensor(buf825, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf825  # reuse
    cpp_fused_clone_177(c_void_p(buf869.data_ptr()), c_void_p(buf870.data_ptr()), c_void_p(buf871.data_ptr()), c_void_p(buf872.data_ptr()), c_void_p(buf873.data_ptr()), c_void_p(buf874.data_ptr()))
    del buf869
    del buf870
    del buf871
    # Source Nodes: [], Original ATen: []
    buf875 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf872, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf873, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), reinterpret_tensor(buf874, (1, 16, 1024, 64), (0, 65536, 64, 1), 0), scale=1.0)
    del buf872
    buf876 = buf875[0]
    del buf875
    buf883 = reinterpret_tensor(buf876, (1, 1024, 16, 64), (1048576, 1024, 64, 1), 0); del buf876  # reuse
    cpp_fused_clone_178(c_void_p(buf883.data_ptr()))
    buf884 = reinterpret_tensor(buf874, (1024, 1024), (1024, 1), 0); del buf874  # reuse
    # Source Nodes: [hidden_states_309], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg507_1, reinterpret_tensor(buf883, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg506_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf884)
    del arg506_1
    del arg507_1
    buf885 = buf866; del buf866  # reuse
    buf886 = buf865; del buf865  # reuse
    buf888 = reinterpret_tensor(buf883, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf883  # reuse
    cpp_fused_add_native_layer_norm_179(c_void_p(buf839.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(buf864.data_ptr()), c_void_p(buf884.data_ptr()), c_void_p(arg508_1.data_ptr()), c_void_p(arg509_1.data_ptr()), c_void_p(buf885.data_ptr()), c_void_p(buf886.data_ptr()), c_void_p(buf888.data_ptr()))
    del arg508_1
    del arg509_1
    buf889 = reinterpret_tensor(buf845, (1024, 4096), (4096, 1), 0); del buf845  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg511_1, reinterpret_tensor(buf888, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg510_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf889)
    del arg510_1
    del arg511_1
    buf890 = reinterpret_tensor(buf889, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf889  # reuse
    cpp_fused_gelu_180(c_void_p(buf890.data_ptr()))
    buf891 = reinterpret_tensor(buf888, (1024, 1024), (1024, 1), 0); del buf888  # reuse
    # Source Nodes: [hidden_states_315], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg513_1, reinterpret_tensor(buf890, (1024, 4096), (4096, 1), 0), reinterpret_tensor(arg512_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf891)
    del arg512_1
    del arg513_1
    del buf890
    buf892 = reinterpret_tensor(buf891, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf891  # reuse
    buf893 = buf886; del buf886  # reuse
    buf894 = buf885; del buf885  # reuse
    buf896 = reinterpret_tensor(buf873, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf873  # reuse
    cpp_fused_add_native_layer_norm_181(c_void_p(buf892.data_ptr()), c_void_p(buf839.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(buf864.data_ptr()), c_void_p(buf884.data_ptr()), c_void_p(arg514_1.data_ptr()), c_void_p(arg515_1.data_ptr()), c_void_p(buf893.data_ptr()), c_void_p(buf894.data_ptr()), c_void_p(buf896.data_ptr()))
    del arg514_1
    del arg515_1
    del buf839
    del buf846
    del buf864
    del buf884
    del buf892
    buf897 = empty((1024, 50265), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___lm_head], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf896, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg516_1, (1024, 50265), (1, 1024), 0), out=buf897)
    del arg516_1
    del buf896
    buf898 = reinterpret_tensor(buf897, (1, 1024, 50265), (51471360, 50265, 1), 0); del buf897  # reuse
    buf899 = reinterpret_tensor(buf894, (1024, 1), (1, 1024), 0); del buf894  # reuse
    buf900 = reinterpret_tensor(buf893, (1024, 1), (1, 1024), 0); del buf893  # reuse
    buf901 = empty((), device='cpu', dtype=torch.float32)
    buf902 = reinterpret_tensor(buf337, (), (), 0); del buf337  # reuse
    buf903 = buf901; del buf901  # reuse
    cpp_fused__log_softmax_add_nll_loss_forward_182(c_void_p(buf898.data_ptr()), c_void_p(buf903.data_ptr()), c_void_p(arg517_1.data_ptr()), c_void_p(arg518_1.data_ptr()), c_void_p(buf899.data_ptr()), c_void_p(buf900.data_ptr()), c_void_p(buf902.data_ptr()))
    del arg517_1
    del arg518_1
    return (buf903, buf898, buf366, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1026, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1026, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((50265, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((50265, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg397_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg400_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg403_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg406_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg409_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg412_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg415_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg416_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg418_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg419_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg421_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg422_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg424_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg425_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg427_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg428_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg430_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg431_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg433_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg434_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg436_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg437_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg439_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg440_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg442_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg443_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg445_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg446_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg448_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg449_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg451_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg452_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg454_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg455_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg457_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg458_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg459_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg460_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg461_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg462_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg463_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg464_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg465_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg466_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg467_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg468_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg469_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg470_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg471_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg472_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg473_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg474_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg475_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg476_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg477_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg478_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg479_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg480_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg481_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg482_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg483_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg484_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg485_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg486_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg487_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg488_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg489_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg490_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg491_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg492_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg493_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg494_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg495_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg496_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg497_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg498_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg499_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg500_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg501_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg502_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg503_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg504_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg505_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg506_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg507_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg508_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg509_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg510_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg511_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg512_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg513_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg514_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg515_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg516_1 = rand_strided((50265, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg517_1 = rand_strided((1, 50265), (50265, 1), device='cpu', dtype=torch.float32)
    arg518_1 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    arg519_1 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MBartForConditionalGeneration', benchmark_compiled_module)
