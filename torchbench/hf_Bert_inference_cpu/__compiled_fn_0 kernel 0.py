
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x1)];
                        auto tmp11 = in_ptr4[static_cast<long>(x1)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 30522);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 30522L), "index out of bounds: 0 <= tmp3 < 30522L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*tmp3)));
                        auto tmp6 = decltype(tmp5)(tmp5 + 2);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 2L), "index out of bounds: 0 <= tmp8 < 2L")
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*tmp8)));
                        auto tmp10 = tmp4 + tmp9;
                        auto tmp12 = decltype(tmp11)(tmp11 + 512);
                        auto tmp13 = tmp11 < 0;
                        auto tmp14 = tmp13 ? tmp12 : tmp11;
                        TORCH_CHECK((0 <= tmp14) & (tmp14 < 512L), "index out of bounds: 0 <= tmp14 < 512L")
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (768L*tmp14)));
                        auto tmp16 = tmp10 + tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (393216L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_gelu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_gelu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_add_native_layer_norm_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_gelu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_gelu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_gelu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_add_native_layer_norm_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_gelu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_gelu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_gelu_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_add_native_layer_norm_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_gelu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_gelu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
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


cpp_fused_gelu_native_layer_norm_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    auto tmp16 = static_cast<float>(768.0);
                    auto tmp17 = tmp15 / tmp16;
                    auto tmp18 = static_cast<float>(1e-12);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = 1 / std::sqrt(tmp19);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp14 * tmp21;
                    auto tmp24 = tmp22 * tmp23;
                    auto tmp26 = tmp24 + tmp25;
                    tmp26.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1 = args
    args.clear()
    assert_size_stride(arg0_1, (30522, 768), (768, 1))
    assert_size_stride(arg1_1, (2, 768), (768, 1))
    assert_size_stride(arg2_1, (512, 768), (768, 1))
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
    assert_size_stride(arg101_1, (768, 768), (768, 1))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, 768), (768, 1))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, 768), (768, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, 768), (768, 1))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (3072, 768), (768, 1))
    assert_size_stride(arg112_1, (3072, ), (1, ))
    assert_size_stride(arg113_1, (768, 3072), (3072, 1))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, 768), (768, 1))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, 768), (768, 1))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, 768), (768, 1))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, 768), (768, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (3072, 768), (768, 1))
    assert_size_stride(arg128_1, (3072, ), (1, ))
    assert_size_stride(arg129_1, (768, 3072), (3072, 1))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, 768), (768, 1))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, 768), (768, 1))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, 768), (768, 1))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, 768), (768, 1))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (3072, 768), (768, 1))
    assert_size_stride(arg144_1, (3072, ), (1, ))
    assert_size_stride(arg145_1, (768, 3072), (3072, 1))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, 768), (768, 1))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (768, 768), (768, 1))
    assert_size_stride(arg152_1, (768, ), (1, ))
    assert_size_stride(arg153_1, (768, 768), (768, 1))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, 768), (768, 1))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (3072, 768), (768, 1))
    assert_size_stride(arg160_1, (3072, ), (1, ))
    assert_size_stride(arg161_1, (768, 3072), (3072, 1))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, 768), (768, 1))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, 768), (768, 1))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, 768), (768, 1))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (768, 768), (768, 1))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (3072, 768), (768, 1))
    assert_size_stride(arg176_1, (3072, ), (1, ))
    assert_size_stride(arg177_1, (768, 3072), (3072, 1))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, 768), (768, 1))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, 768), (768, 1))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, 768), (768, 1))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, 768), (768, 1))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (3072, 768), (768, 1))
    assert_size_stride(arg192_1, (3072, ), (1, ))
    assert_size_stride(arg193_1, (768, 3072), (3072, 1))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, 768), (768, 1))
    assert_size_stride(arg198_1, (768, ), (1, ))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (30522, 768), (768, 1))
    assert_size_stride(arg202_1, (30522, ), (1, ))
    assert_size_stride(arg203_1, (1, 512), (512, 1))
    assert_size_stride(arg204_1, (1, 512), (512, 1))
    assert_size_stride(arg205_1, (4, 512), (512, 1))
    buf0 = empty((4, 512, 768), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((4, 512, 1), (512, 1, 2048), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((4, 512, 1), (512, 1, 2048), device='cpu', dtype=torch.float32)
    buf4 = empty((4, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_0(c_void_p(arg205_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg0_1
    del arg1_1
    del arg203_1
    del arg204_1
    del arg205_1
    del arg2_1
    del arg3_1
    del arg4_1
    buf5 = reinterpret_tensor(buf0, (2048, 768), (768, 1), 0); del buf0  # reuse
    # Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg6_1, reinterpret_tensor(buf4, (2048, 768), (768, 1), 0), reinterpret_tensor(arg5_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf5)
    del arg5_1
    del arg6_1
    buf6 = empty((2048, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___bert_encoder_layer_0_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf4, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
    del arg7_1
    del arg8_1
    buf7 = empty((2048, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___bert_encoder_layer_0_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf4, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
    del arg10_1
    del arg9_1
    # Source Nodes: [], Original ATen: []
    buf8 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf5, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf6, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf7, (4, 12, 512, 64), (393216, 64, 768, 1), 0), scale=0.125)
    del buf5
    buf9 = buf8[0]
    del buf8
    buf16 = buf7; del buf7  # reuse
    # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf9, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf16)
    del arg11_1
    del arg12_1
    buf17 = buf2; del buf2  # reuse
    buf18 = buf1; del buf1  # reuse
    buf20 = reinterpret_tensor(buf9, (4, 512, 768), (393216, 768, 1), 0); del buf9  # reuse
    cpp_fused_add_native_layer_norm_1(c_void_p(buf16.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()))
    del arg13_1
    del arg14_1
    buf21 = empty((2048, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg16_1, reinterpret_tensor(buf20, (2048, 768), (768, 1), 0), reinterpret_tensor(arg15_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf21)
    del arg15_1
    del arg16_1
    buf22 = reinterpret_tensor(buf21, (4, 512, 3072), (1572864, 3072, 1), 0); del buf21  # reuse
    cpp_fused_gelu_2(c_void_p(buf22.data_ptr()))
    buf23 = reinterpret_tensor(buf4, (2048, 768), (768, 1), 0); del buf4  # reuse
    # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf22, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg17_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf23)
    del arg17_1
    del arg18_1
    buf24 = buf18; del buf18  # reuse
    buf25 = buf17; del buf17  # reuse
    buf27 = reinterpret_tensor(buf16, (4, 512, 768), (393216, 768, 1), 0); del buf16  # reuse
    cpp_fused_add_native_layer_norm_3(c_void_p(buf23.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()))
    del arg19_1
    del arg20_1
    buf28 = buf23; del buf23  # reuse
    # Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg22_1, reinterpret_tensor(buf27, (2048, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf28)
    del arg21_1
    del arg22_1
    buf29 = reinterpret_tensor(buf20, (2048, 768), (768, 1), 0); del buf20  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_1_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg24_1, reinterpret_tensor(buf27, (2048, 768), (768, 1), 0), reinterpret_tensor(arg23_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf29)
    del arg23_1
    del arg24_1
    buf30 = buf6; del buf6  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_1_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg26_1, reinterpret_tensor(buf27, (2048, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf30)
    del arg25_1
    del arg26_1
    # Source Nodes: [], Original ATen: []
    buf31 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf28, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf29, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf30, (4, 12, 512, 64), (393216, 64, 768, 1), 0), scale=0.125)
    del buf28
    buf32 = buf31[0]
    del buf31
    buf39 = buf30; del buf30  # reuse
    # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg28_1, reinterpret_tensor(buf32, (2048, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf39)
    del arg27_1
    del arg28_1
    buf40 = buf25; del buf25  # reuse
    buf41 = buf24; del buf24  # reuse
    buf43 = reinterpret_tensor(buf32, (4, 512, 768), (393216, 768, 1), 0); del buf32  # reuse
    cpp_fused_add_native_layer_norm_4(c_void_p(buf39.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()))
    del arg29_1
    del arg30_1
    buf44 = reinterpret_tensor(buf22, (2048, 3072), (3072, 1), 0); del buf22  # reuse
    # Source Nodes: [hidden_states_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg32_1, reinterpret_tensor(buf43, (2048, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf44)
    del arg31_1
    del arg32_1
    buf45 = reinterpret_tensor(buf44, (4, 512, 3072), (1572864, 3072, 1), 0); del buf44  # reuse
    cpp_fused_gelu_5(c_void_p(buf45.data_ptr()))
    buf46 = buf39; del buf39  # reuse
    # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg34_1, reinterpret_tensor(buf45, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg33_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf46)
    del arg33_1
    del arg34_1
    buf47 = buf41; del buf41  # reuse
    buf48 = buf40; del buf40  # reuse
    buf50 = buf27; del buf27  # reuse
    cpp_fused_add_native_layer_norm_6(c_void_p(buf46.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()))
    del arg35_1
    del arg36_1
    buf51 = buf46; del buf46  # reuse
    # Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg38_1, reinterpret_tensor(buf50, (2048, 768), (768, 1), 0), reinterpret_tensor(arg37_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf51)
    del arg37_1
    del arg38_1
    buf52 = reinterpret_tensor(buf43, (2048, 768), (768, 1), 0); del buf43  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_2_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg40_1, reinterpret_tensor(buf50, (2048, 768), (768, 1), 0), reinterpret_tensor(arg39_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf52)
    del arg39_1
    del arg40_1
    buf53 = buf29; del buf29  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_2_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg42_1, reinterpret_tensor(buf50, (2048, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf53)
    del arg41_1
    del arg42_1
    # Source Nodes: [], Original ATen: []
    buf54 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf51, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf52, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf53, (4, 12, 512, 64), (393216, 64, 768, 1), 0), scale=0.125)
    del buf51
    buf55 = buf54[0]
    del buf54
    buf62 = buf53; del buf53  # reuse
    # Source Nodes: [hidden_states_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg44_1, reinterpret_tensor(buf55, (2048, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf62)
    del arg43_1
    del arg44_1
    buf63 = buf48; del buf48  # reuse
    buf64 = buf47; del buf47  # reuse
    buf66 = reinterpret_tensor(buf55, (4, 512, 768), (393216, 768, 1), 0); del buf55  # reuse
    cpp_fused_add_native_layer_norm_7(c_void_p(buf62.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()))
    del arg45_1
    del arg46_1
    buf67 = reinterpret_tensor(buf45, (2048, 3072), (3072, 1), 0); del buf45  # reuse
    # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg48_1, reinterpret_tensor(buf66, (2048, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf67)
    del arg47_1
    del arg48_1
    buf68 = reinterpret_tensor(buf67, (4, 512, 3072), (1572864, 3072, 1), 0); del buf67  # reuse
    cpp_fused_gelu_8(c_void_p(buf68.data_ptr()))
    buf69 = buf62; del buf62  # reuse
    # Source Nodes: [hidden_states_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg50_1, reinterpret_tensor(buf68, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg49_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf69)
    del arg49_1
    del arg50_1
    buf70 = buf64; del buf64  # reuse
    buf71 = buf63; del buf63  # reuse
    buf73 = buf50; del buf50  # reuse
    cpp_fused_add_native_layer_norm_9(c_void_p(buf69.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf73.data_ptr()))
    del arg51_1
    del arg52_1
    buf74 = buf69; del buf69  # reuse
    # Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg54_1, reinterpret_tensor(buf73, (2048, 768), (768, 1), 0), reinterpret_tensor(arg53_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf74)
    del arg53_1
    del arg54_1
    buf75 = reinterpret_tensor(buf66, (2048, 768), (768, 1), 0); del buf66  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_3_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg56_1, reinterpret_tensor(buf73, (2048, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf75)
    del arg55_1
    del arg56_1
    buf76 = buf52; del buf52  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_3_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg58_1, reinterpret_tensor(buf73, (2048, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf76)
    del arg57_1
    del arg58_1
    # Source Nodes: [], Original ATen: []
    buf77 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf74, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf75, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf76, (4, 12, 512, 64), (393216, 64, 768, 1), 0), scale=0.125)
    del buf74
    buf78 = buf77[0]
    del buf77
    buf85 = buf76; del buf76  # reuse
    # Source Nodes: [hidden_states_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg60_1, reinterpret_tensor(buf78, (2048, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf85)
    del arg59_1
    del arg60_1
    buf86 = buf71; del buf71  # reuse
    buf87 = buf70; del buf70  # reuse
    buf89 = reinterpret_tensor(buf78, (4, 512, 768), (393216, 768, 1), 0); del buf78  # reuse
    cpp_fused_add_native_layer_norm_10(c_void_p(buf85.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()))
    del arg61_1
    del arg62_1
    buf90 = reinterpret_tensor(buf68, (2048, 3072), (3072, 1), 0); del buf68  # reuse
    # Source Nodes: [hidden_states_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg64_1, reinterpret_tensor(buf89, (2048, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf90)
    del arg63_1
    del arg64_1
    buf91 = reinterpret_tensor(buf90, (4, 512, 3072), (1572864, 3072, 1), 0); del buf90  # reuse
    cpp_fused_gelu_11(c_void_p(buf91.data_ptr()))
    buf92 = buf85; del buf85  # reuse
    # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg66_1, reinterpret_tensor(buf91, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg65_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf92)
    del arg65_1
    del arg66_1
    buf93 = buf87; del buf87  # reuse
    buf94 = buf86; del buf86  # reuse
    buf96 = buf73; del buf73  # reuse
    cpp_fused_add_native_layer_norm_12(c_void_p(buf92.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()))
    del arg67_1
    del arg68_1
    buf97 = buf92; del buf92  # reuse
    # Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg70_1, reinterpret_tensor(buf96, (2048, 768), (768, 1), 0), reinterpret_tensor(arg69_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf97)
    del arg69_1
    del arg70_1
    buf98 = reinterpret_tensor(buf89, (2048, 768), (768, 1), 0); del buf89  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_4_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg72_1, reinterpret_tensor(buf96, (2048, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf98)
    del arg71_1
    del arg72_1
    buf99 = buf75; del buf75  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_4_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg74_1, reinterpret_tensor(buf96, (2048, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf99)
    del arg73_1
    del arg74_1
    # Source Nodes: [], Original ATen: []
    buf100 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf97, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf98, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf99, (4, 12, 512, 64), (393216, 64, 768, 1), 0), scale=0.125)
    del buf97
    buf101 = buf100[0]
    del buf100
    buf108 = buf99; del buf99  # reuse
    # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg76_1, reinterpret_tensor(buf101, (2048, 768), (768, 1), 0), reinterpret_tensor(arg75_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf108)
    del arg75_1
    del arg76_1
    buf109 = buf94; del buf94  # reuse
    buf110 = buf93; del buf93  # reuse
    buf112 = reinterpret_tensor(buf101, (4, 512, 768), (393216, 768, 1), 0); del buf101  # reuse
    cpp_fused_add_native_layer_norm_13(c_void_p(buf108.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf112.data_ptr()))
    del arg77_1
    del arg78_1
    buf113 = reinterpret_tensor(buf91, (2048, 3072), (3072, 1), 0); del buf91  # reuse
    # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg80_1, reinterpret_tensor(buf112, (2048, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf113)
    del arg79_1
    del arg80_1
    buf114 = reinterpret_tensor(buf113, (4, 512, 3072), (1572864, 3072, 1), 0); del buf113  # reuse
    cpp_fused_gelu_14(c_void_p(buf114.data_ptr()))
    buf115 = reinterpret_tensor(buf96, (2048, 768), (768, 1), 0); del buf96  # reuse
    # Source Nodes: [hidden_states_41], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg82_1, reinterpret_tensor(buf114, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg81_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf115)
    del arg81_1
    del arg82_1
    buf116 = buf110; del buf110  # reuse
    buf117 = buf109; del buf109  # reuse
    buf119 = reinterpret_tensor(buf108, (4, 512, 768), (393216, 768, 1), 0); del buf108  # reuse
    cpp_fused_add_native_layer_norm_15(c_void_p(buf115.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()))
    del arg83_1
    del arg84_1
    buf120 = buf115; del buf115  # reuse
    # Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg86_1, reinterpret_tensor(buf119, (2048, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf120)
    del arg85_1
    del arg86_1
    buf121 = reinterpret_tensor(buf112, (2048, 768), (768, 1), 0); del buf112  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_5_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg88_1, reinterpret_tensor(buf119, (2048, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf121)
    del arg87_1
    del arg88_1
    buf122 = buf98; del buf98  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_5_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg90_1, reinterpret_tensor(buf119, (2048, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf122)
    del arg89_1
    del arg90_1
    # Source Nodes: [], Original ATen: []
    buf123 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf120, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf121, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf122, (4, 12, 512, 64), (393216, 64, 768, 1), 0), scale=0.125)
    del buf120
    buf124 = buf123[0]
    del buf123
    buf131 = buf122; del buf122  # reuse
    # Source Nodes: [hidden_states_45], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg92_1, reinterpret_tensor(buf124, (2048, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf131)
    del arg91_1
    del arg92_1
    buf132 = buf117; del buf117  # reuse
    buf133 = buf116; del buf116  # reuse
    buf135 = reinterpret_tensor(buf124, (4, 512, 768), (393216, 768, 1), 0); del buf124  # reuse
    cpp_fused_add_native_layer_norm_16(c_void_p(buf131.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()))
    del arg93_1
    del arg94_1
    buf136 = reinterpret_tensor(buf114, (2048, 3072), (3072, 1), 0); del buf114  # reuse
    # Source Nodes: [hidden_states_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg96_1, reinterpret_tensor(buf135, (2048, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf136)
    del arg95_1
    del arg96_1
    buf137 = reinterpret_tensor(buf136, (4, 512, 3072), (1572864, 3072, 1), 0); del buf136  # reuse
    cpp_fused_gelu_17(c_void_p(buf137.data_ptr()))
    buf138 = buf131; del buf131  # reuse
    # Source Nodes: [hidden_states_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg98_1, reinterpret_tensor(buf137, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg97_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf138)
    del arg97_1
    del arg98_1
    buf139 = buf133; del buf133  # reuse
    buf140 = buf132; del buf132  # reuse
    buf142 = buf119; del buf119  # reuse
    cpp_fused_add_native_layer_norm_18(c_void_p(buf138.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf142.data_ptr()))
    del arg100_1
    del arg99_1
    buf143 = buf138; del buf138  # reuse
    # Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg102_1, reinterpret_tensor(buf142, (2048, 768), (768, 1), 0), reinterpret_tensor(arg101_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf143)
    del arg101_1
    del arg102_1
    buf144 = reinterpret_tensor(buf135, (2048, 768), (768, 1), 0); del buf135  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_6_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg104_1, reinterpret_tensor(buf142, (2048, 768), (768, 1), 0), reinterpret_tensor(arg103_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf144)
    del arg103_1
    del arg104_1
    buf145 = buf121; del buf121  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_6_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg106_1, reinterpret_tensor(buf142, (2048, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf145)
    del arg105_1
    del arg106_1
    # Source Nodes: [], Original ATen: []
    buf146 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf143, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf144, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf145, (4, 12, 512, 64), (393216, 64, 768, 1), 0), scale=0.125)
    del buf143
    buf147 = buf146[0]
    del buf146
    buf154 = buf145; del buf145  # reuse
    # Source Nodes: [hidden_states_54], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg108_1, reinterpret_tensor(buf147, (2048, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf154)
    del arg107_1
    del arg108_1
    buf155 = buf140; del buf140  # reuse
    buf156 = buf139; del buf139  # reuse
    buf158 = reinterpret_tensor(buf147, (4, 512, 768), (393216, 768, 1), 0); del buf147  # reuse
    cpp_fused_add_native_layer_norm_19(c_void_p(buf154.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf158.data_ptr()))
    del arg109_1
    del arg110_1
    buf159 = reinterpret_tensor(buf137, (2048, 3072), (3072, 1), 0); del buf137  # reuse
    # Source Nodes: [hidden_states_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg112_1, reinterpret_tensor(buf158, (2048, 768), (768, 1), 0), reinterpret_tensor(arg111_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf159)
    del arg111_1
    del arg112_1
    buf160 = reinterpret_tensor(buf159, (4, 512, 3072), (1572864, 3072, 1), 0); del buf159  # reuse
    cpp_fused_gelu_20(c_void_p(buf160.data_ptr()))
    buf161 = buf154; del buf154  # reuse
    # Source Nodes: [hidden_states_59], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg114_1, reinterpret_tensor(buf160, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg113_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf161)
    del arg113_1
    del arg114_1
    buf162 = buf156; del buf156  # reuse
    buf163 = buf155; del buf155  # reuse
    buf165 = buf142; del buf142  # reuse
    cpp_fused_add_native_layer_norm_21(c_void_p(buf161.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf165.data_ptr()))
    del arg115_1
    del arg116_1
    buf166 = buf161; del buf161  # reuse
    # Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg118_1, reinterpret_tensor(buf165, (2048, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf166)
    del arg117_1
    del arg118_1
    buf167 = reinterpret_tensor(buf158, (2048, 768), (768, 1), 0); del buf158  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_7_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg120_1, reinterpret_tensor(buf165, (2048, 768), (768, 1), 0), reinterpret_tensor(arg119_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf167)
    del arg119_1
    del arg120_1
    buf168 = buf144; del buf144  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_7_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg122_1, reinterpret_tensor(buf165, (2048, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf168)
    del arg121_1
    del arg122_1
    # Source Nodes: [], Original ATen: []
    buf169 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf166, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf167, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf168, (4, 12, 512, 64), (393216, 64, 768, 1), 0), scale=0.125)
    del buf166
    buf170 = buf169[0]
    del buf169
    buf177 = buf168; del buf168  # reuse
    # Source Nodes: [hidden_states_63], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg124_1, reinterpret_tensor(buf170, (2048, 768), (768, 1), 0), reinterpret_tensor(arg123_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf177)
    del arg123_1
    del arg124_1
    buf178 = buf163; del buf163  # reuse
    buf179 = buf162; del buf162  # reuse
    buf181 = reinterpret_tensor(buf170, (4, 512, 768), (393216, 768, 1), 0); del buf170  # reuse
    cpp_fused_add_native_layer_norm_22(c_void_p(buf177.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()))
    del arg125_1
    del arg126_1
    buf182 = reinterpret_tensor(buf160, (2048, 3072), (3072, 1), 0); del buf160  # reuse
    # Source Nodes: [hidden_states_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg128_1, reinterpret_tensor(buf181, (2048, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf182)
    del arg127_1
    del arg128_1
    buf183 = reinterpret_tensor(buf182, (4, 512, 3072), (1572864, 3072, 1), 0); del buf182  # reuse
    cpp_fused_gelu_23(c_void_p(buf183.data_ptr()))
    buf184 = buf177; del buf177  # reuse
    # Source Nodes: [hidden_states_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg130_1, reinterpret_tensor(buf183, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg129_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf184)
    del arg129_1
    del arg130_1
    buf185 = buf179; del buf179  # reuse
    buf186 = buf178; del buf178  # reuse
    buf188 = buf165; del buf165  # reuse
    cpp_fused_add_native_layer_norm_24(c_void_p(buf184.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf188.data_ptr()))
    del arg131_1
    del arg132_1
    buf189 = buf184; del buf184  # reuse
    # Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg134_1, reinterpret_tensor(buf188, (2048, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf189)
    del arg133_1
    del arg134_1
    buf190 = reinterpret_tensor(buf181, (2048, 768), (768, 1), 0); del buf181  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_8_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg136_1, reinterpret_tensor(buf188, (2048, 768), (768, 1), 0), reinterpret_tensor(arg135_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf190)
    del arg135_1
    del arg136_1
    buf191 = buf167; del buf167  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_8_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf188, (2048, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf191)
    del arg137_1
    del arg138_1
    # Source Nodes: [], Original ATen: []
    buf192 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf189, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf190, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf191, (4, 12, 512, 64), (393216, 64, 768, 1), 0), scale=0.125)
    del buf189
    buf193 = buf192[0]
    del buf192
    buf200 = buf191; del buf191  # reuse
    # Source Nodes: [hidden_states_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg140_1, reinterpret_tensor(buf193, (2048, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf200)
    del arg139_1
    del arg140_1
    buf201 = buf186; del buf186  # reuse
    buf202 = buf185; del buf185  # reuse
    buf204 = reinterpret_tensor(buf193, (4, 512, 768), (393216, 768, 1), 0); del buf193  # reuse
    cpp_fused_add_native_layer_norm_25(c_void_p(buf200.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()))
    del arg141_1
    del arg142_1
    buf205 = reinterpret_tensor(buf183, (2048, 3072), (3072, 1), 0); del buf183  # reuse
    # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg144_1, reinterpret_tensor(buf204, (2048, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf205)
    del arg143_1
    del arg144_1
    buf206 = reinterpret_tensor(buf205, (4, 512, 3072), (1572864, 3072, 1), 0); del buf205  # reuse
    cpp_fused_gelu_26(c_void_p(buf206.data_ptr()))
    buf207 = buf200; del buf200  # reuse
    # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg146_1, reinterpret_tensor(buf206, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg145_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf207)
    del arg145_1
    del arg146_1
    buf208 = buf202; del buf202  # reuse
    buf209 = buf201; del buf201  # reuse
    buf211 = buf188; del buf188  # reuse
    cpp_fused_add_native_layer_norm_27(c_void_p(buf207.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()))
    del arg147_1
    del arg148_1
    buf212 = buf207; del buf207  # reuse
    # Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg150_1, reinterpret_tensor(buf211, (2048, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf212)
    del arg149_1
    del arg150_1
    buf213 = reinterpret_tensor(buf204, (2048, 768), (768, 1), 0); del buf204  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_9_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg152_1, reinterpret_tensor(buf211, (2048, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf213)
    del arg151_1
    del arg152_1
    buf214 = buf190; del buf190  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_9_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg154_1, reinterpret_tensor(buf211, (2048, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf214)
    del arg153_1
    del arg154_1
    # Source Nodes: [], Original ATen: []
    buf215 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf212, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf213, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf214, (4, 12, 512, 64), (393216, 64, 768, 1), 0), scale=0.125)
    del buf212
    buf216 = buf215[0]
    del buf215
    buf223 = buf214; del buf214  # reuse
    # Source Nodes: [hidden_states_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg156_1, reinterpret_tensor(buf216, (2048, 768), (768, 1), 0), reinterpret_tensor(arg155_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf223)
    del arg155_1
    del arg156_1
    buf224 = buf209; del buf209  # reuse
    buf225 = buf208; del buf208  # reuse
    buf227 = reinterpret_tensor(buf216, (4, 512, 768), (393216, 768, 1), 0); del buf216  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf223.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf227.data_ptr()))
    del arg157_1
    del arg158_1
    buf228 = reinterpret_tensor(buf206, (2048, 3072), (3072, 1), 0); del buf206  # reuse
    # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg160_1, reinterpret_tensor(buf227, (2048, 768), (768, 1), 0), reinterpret_tensor(arg159_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf228)
    del arg159_1
    del arg160_1
    buf229 = reinterpret_tensor(buf228, (4, 512, 3072), (1572864, 3072, 1), 0); del buf228  # reuse
    cpp_fused_gelu_29(c_void_p(buf229.data_ptr()))
    buf230 = buf223; del buf223  # reuse
    # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg162_1, reinterpret_tensor(buf229, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg161_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf230)
    del arg161_1
    del arg162_1
    buf231 = buf225; del buf225  # reuse
    buf232 = buf224; del buf224  # reuse
    buf234 = buf211; del buf211  # reuse
    cpp_fused_add_native_layer_norm_30(c_void_p(buf230.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()))
    del arg163_1
    del arg164_1
    buf235 = buf230; del buf230  # reuse
    # Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg166_1, reinterpret_tensor(buf234, (2048, 768), (768, 1), 0), reinterpret_tensor(arg165_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf235)
    del arg165_1
    del arg166_1
    buf236 = reinterpret_tensor(buf227, (2048, 768), (768, 1), 0); del buf227  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_10_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg168_1, reinterpret_tensor(buf234, (2048, 768), (768, 1), 0), reinterpret_tensor(arg167_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf236)
    del arg167_1
    del arg168_1
    buf237 = buf213; del buf213  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_10_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg170_1, reinterpret_tensor(buf234, (2048, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf237)
    del arg169_1
    del arg170_1
    # Source Nodes: [], Original ATen: []
    buf238 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf235, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf236, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf237, (4, 12, 512, 64), (393216, 64, 768, 1), 0), scale=0.125)
    del buf235
    buf239 = buf238[0]
    del buf238
    buf246 = buf237; del buf237  # reuse
    # Source Nodes: [hidden_states_90], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg172_1, reinterpret_tensor(buf239, (2048, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf246)
    del arg171_1
    del arg172_1
    buf247 = buf232; del buf232  # reuse
    buf248 = buf231; del buf231  # reuse
    buf250 = reinterpret_tensor(buf239, (4, 512, 768), (393216, 768, 1), 0); del buf239  # reuse
    cpp_fused_add_native_layer_norm_31(c_void_p(buf246.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf250.data_ptr()))
    del arg173_1
    del arg174_1
    buf251 = reinterpret_tensor(buf229, (2048, 3072), (3072, 1), 0); del buf229  # reuse
    # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg176_1, reinterpret_tensor(buf250, (2048, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf251)
    del arg175_1
    del arg176_1
    buf252 = reinterpret_tensor(buf251, (4, 512, 3072), (1572864, 3072, 1), 0); del buf251  # reuse
    cpp_fused_gelu_32(c_void_p(buf252.data_ptr()))
    buf253 = buf246; del buf246  # reuse
    # Source Nodes: [hidden_states_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg178_1, reinterpret_tensor(buf252, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg177_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf253)
    del arg177_1
    del arg178_1
    buf254 = buf248; del buf248  # reuse
    buf255 = buf247; del buf247  # reuse
    buf257 = buf234; del buf234  # reuse
    cpp_fused_add_native_layer_norm_33(c_void_p(buf253.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf257.data_ptr()))
    del arg179_1
    del arg180_1
    buf258 = buf253; del buf253  # reuse
    # Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg182_1, reinterpret_tensor(buf257, (2048, 768), (768, 1), 0), reinterpret_tensor(arg181_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf258)
    del arg181_1
    del arg182_1
    buf259 = reinterpret_tensor(buf250, (2048, 768), (768, 1), 0); del buf250  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_11_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg184_1, reinterpret_tensor(buf257, (2048, 768), (768, 1), 0), reinterpret_tensor(arg183_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf259)
    del arg183_1
    del arg184_1
    buf260 = buf236; del buf236  # reuse
    # Source Nodes: [l__mod___bert_encoder_layer_11_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg186_1, reinterpret_tensor(buf257, (2048, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf260)
    del arg185_1
    del arg186_1
    # Source Nodes: [], Original ATen: []
    buf261 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf258, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf259, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf260, (4, 12, 512, 64), (393216, 64, 768, 1), 0), scale=0.125)
    del buf258
    del buf259
    buf262 = buf261[0]
    del buf261
    buf269 = buf260; del buf260  # reuse
    # Source Nodes: [hidden_states_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg188_1, reinterpret_tensor(buf262, (2048, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf269)
    del arg187_1
    del arg188_1
    buf270 = buf255; del buf255  # reuse
    buf271 = buf254; del buf254  # reuse
    buf273 = reinterpret_tensor(buf262, (4, 512, 768), (393216, 768, 1), 0); del buf262  # reuse
    cpp_fused_add_native_layer_norm_34(c_void_p(buf269.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()))
    del arg189_1
    del arg190_1
    buf274 = reinterpret_tensor(buf252, (2048, 3072), (3072, 1), 0); del buf252  # reuse
    # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg192_1, reinterpret_tensor(buf273, (2048, 768), (768, 1), 0), reinterpret_tensor(arg191_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf274)
    del arg191_1
    del arg192_1
    buf275 = reinterpret_tensor(buf274, (4, 512, 3072), (1572864, 3072, 1), 0); del buf274  # reuse
    cpp_fused_gelu_35(c_void_p(buf275.data_ptr()))
    buf276 = buf269; del buf269  # reuse
    # Source Nodes: [hidden_states_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg194_1, reinterpret_tensor(buf275, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg193_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf276)
    del arg193_1
    del arg194_1
    del buf275
    buf277 = buf271; del buf271  # reuse
    buf278 = buf270; del buf270  # reuse
    buf280 = buf257; del buf257  # reuse
    cpp_fused_add_native_layer_norm_36(c_void_p(buf276.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf280.data_ptr()))
    del arg195_1
    del arg196_1
    del buf273
    buf281 = buf276; del buf276  # reuse
    # Source Nodes: [hidden_states_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg198_1, reinterpret_tensor(buf280, (2048, 768), (768, 1), 0), reinterpret_tensor(arg197_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf281)
    del arg197_1
    del arg198_1
    buf282 = buf278; del buf278  # reuse
    buf283 = buf277; del buf277  # reuse
    buf285 = buf280; del buf280  # reuse
    cpp_fused_gelu_native_layer_norm_37(c_void_p(buf281.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf285.data_ptr()))
    del arg199_1
    del arg200_1
    del buf281
    del buf282
    del buf283
    buf286 = empty((2048, 30522), device='cpu', dtype=torch.float32)
    # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg202_1, reinterpret_tensor(buf285, (2048, 768), (768, 1), 0), reinterpret_tensor(arg201_1, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf286)
    del arg201_1
    del arg202_1
    return (reinterpret_tensor(buf286, (4, 512, 30522), (15627264, 30522, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
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
    arg101_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((30522, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg204_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg205_1 = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_Bert', benchmark_compiled_module)
