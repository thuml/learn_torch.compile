
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
                    auto tmp1 = decltype(tmp0)(tmp0 + 30522);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 30522L), "index out of bounds: 0 <= tmp3 < 30522L")
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_add_clamp_clone_div_nll_loss_forward_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const long* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr1 = in_out_ptr0;
    {
        {
            #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((2L*x0) + (2L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
            out_ptr1[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = out_ptr1[static_cast<long>(0L)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp4 = tmp3.exp();
                tmp_acc0_vec = tmp_acc0_vec + tmp4;
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr2[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(1L + (2L*x0) + (2L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
            out_ptr4[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = out_ptr4[static_cast<long>(0L)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp4 = tmp3.exp();
                tmp_acc0_vec = tmp_acc0_vec + tmp4;
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr5[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        auto tmp0 = in_ptr1[static_cast<long>(0L)];
        auto tmp11 = out_ptr1[static_cast<long>(0L)];
        auto tmp13 = out_ptr2[static_cast<long>(0L)];
        auto tmp22 = in_ptr2[static_cast<long>(0L)];
        auto tmp31 = out_ptr4[static_cast<long>(0L)];
        auto tmp33 = out_ptr5[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(0);
        auto tmp2 = max_propagate_nan(tmp0, tmp1);
        auto tmp3 = static_cast<long>(512);
        auto tmp4 = min_propagate_nan(tmp2, tmp3);
        auto tmp5 = tmp4 != tmp3;
        auto tmp6 = tmp5 ? tmp4 : tmp1;
        auto tmp7 = decltype(tmp6)(tmp6 + 512);
        auto tmp8 = tmp6 < 0;
        auto tmp9 = tmp8 ? tmp7 : tmp6;
        TORCH_CHECK((0 <= tmp9) & (tmp9 < 512L), "index out of bounds: 0 <= tmp9 < 512L")
        auto tmp10 = out_ptr0[static_cast<long>(tmp9)];
        auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
        auto tmp14 = std::log(tmp13);
        auto tmp15 = decltype(tmp12)(tmp12 - tmp14);
        auto tmp16 = decltype(tmp15)(-tmp15);
        auto tmp17 = static_cast<float>(0.0);
        auto tmp18 = tmp5 ? tmp16 : tmp17;
        auto tmp19 = c10::convert<long>(tmp5);
        auto tmp20 = c10::convert<float>(tmp19);
        auto tmp21 = tmp18 / tmp20;
        auto tmp23 = max_propagate_nan(tmp22, tmp1);
        auto tmp24 = min_propagate_nan(tmp23, tmp3);
        auto tmp25 = tmp24 != tmp3;
        auto tmp26 = tmp25 ? tmp24 : tmp1;
        auto tmp27 = decltype(tmp26)(tmp26 + 512);
        auto tmp28 = tmp26 < 0;
        auto tmp29 = tmp28 ? tmp27 : tmp26;
        TORCH_CHECK((0 <= tmp29) & (tmp29 < 512L), "index out of bounds: 0 <= tmp29 < 512L")
        auto tmp30 = out_ptr3[static_cast<long>(tmp29)];
        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
        auto tmp34 = std::log(tmp33);
        auto tmp35 = decltype(tmp32)(tmp32 - tmp34);
        auto tmp36 = decltype(tmp35)(-tmp35);
        auto tmp37 = tmp25 ? tmp36 : tmp17;
        auto tmp38 = c10::convert<long>(tmp25);
        auto tmp39 = c10::convert<float>(tmp38);
        auto tmp40 = tmp37 / tmp39;
        auto tmp41 = decltype(tmp21)(tmp21 + tmp40);
        auto tmp42 = static_cast<float>(2.0);
        auto tmp43 = tmp41 / tmp42;
        in_out_ptr0[static_cast<long>(0L)] = tmp43;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1 = args
    args.clear()
    assert_size_stride(arg0_1, (30522, 128), (128, 1))
    assert_size_stride(arg1_1, (2, 128), (128, 1))
    assert_size_stride(arg2_1, (512, 128), (128, 1))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (256, 128), (128, 1))
    assert_size_stride(arg6_1, (256, ), (1, ))
    assert_size_stride(arg7_1, (256, 256), (256, 1))
    assert_size_stride(arg8_1, (256, ), (1, ))
    assert_size_stride(arg9_1, (256, 256), (256, 1))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, 256), (256, 1))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, 256), (256, 1))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (1024, 256), (256, 1))
    assert_size_stride(arg18_1, (1024, ), (1, ))
    assert_size_stride(arg19_1, (256, 1024), (1024, 1))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, 256), (256, 1))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, 256), (256, 1))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (256, 256), (256, 1))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, 256), (256, 1))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (1024, 256), (256, 1))
    assert_size_stride(arg34_1, (1024, ), (1, ))
    assert_size_stride(arg35_1, (256, 1024), (1024, 1))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, 256), (256, 1))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, 256), (256, 1))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, 256), (256, 1))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, 256), (256, 1))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (1024, 256), (256, 1))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (256, 1024), (1024, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (256, ), (1, ))
    assert_size_stride(arg55_1, (256, 256), (256, 1))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (256, 256), (256, 1))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (256, 256), (256, 1))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (256, 256), (256, 1))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (256, ), (1, ))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (1024, 256), (256, 1))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (256, 1024), (1024, 1))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (256, ), (1, ))
    assert_size_stride(arg70_1, (256, ), (1, ))
    assert_size_stride(arg71_1, (256, 256), (256, 1))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, 256), (256, 1))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, 256), (256, 1))
    assert_size_stride(arg76_1, (256, ), (1, ))
    assert_size_stride(arg77_1, (256, 256), (256, 1))
    assert_size_stride(arg78_1, (256, ), (1, ))
    assert_size_stride(arg79_1, (256, ), (1, ))
    assert_size_stride(arg80_1, (256, ), (1, ))
    assert_size_stride(arg81_1, (1024, 256), (256, 1))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (256, 1024), (1024, 1))
    assert_size_stride(arg84_1, (256, ), (1, ))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (256, 256), (256, 1))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, 256), (256, 1))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (256, 256), (256, 1))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (256, 256), (256, 1))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (256, ), (1, ))
    assert_size_stride(arg97_1, (1024, 256), (256, 1))
    assert_size_stride(arg98_1, (1024, ), (1, ))
    assert_size_stride(arg99_1, (256, 1024), (1024, 1))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (256, 256), (256, 1))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, 256), (256, 1))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (256, 256), (256, 1))
    assert_size_stride(arg108_1, (256, ), (1, ))
    assert_size_stride(arg109_1, (256, 256), (256, 1))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (1024, 256), (256, 1))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (256, 1024), (1024, 1))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (256, ), (1, ))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (256, 256), (256, 1))
    assert_size_stride(arg120_1, (256, ), (1, ))
    assert_size_stride(arg121_1, (256, 256), (256, 1))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, 256), (256, 1))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (256, 256), (256, 1))
    assert_size_stride(arg126_1, (256, ), (1, ))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (1024, 256), (256, 1))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (256, 1024), (1024, 1))
    assert_size_stride(arg132_1, (256, ), (1, ))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (256, ), (1, ))
    assert_size_stride(arg135_1, (256, 256), (256, 1))
    assert_size_stride(arg136_1, (256, ), (1, ))
    assert_size_stride(arg137_1, (256, 256), (256, 1))
    assert_size_stride(arg138_1, (256, ), (1, ))
    assert_size_stride(arg139_1, (256, 256), (256, 1))
    assert_size_stride(arg140_1, (256, ), (1, ))
    assert_size_stride(arg141_1, (256, 256), (256, 1))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (1024, 256), (256, 1))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (256, 1024), (1024, 1))
    assert_size_stride(arg148_1, (256, ), (1, ))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (256, ), (1, ))
    assert_size_stride(arg151_1, (256, 256), (256, 1))
    assert_size_stride(arg152_1, (256, ), (1, ))
    assert_size_stride(arg153_1, (256, 256), (256, 1))
    assert_size_stride(arg154_1, (256, ), (1, ))
    assert_size_stride(arg155_1, (256, 256), (256, 1))
    assert_size_stride(arg156_1, (256, ), (1, ))
    assert_size_stride(arg157_1, (256, 256), (256, 1))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (256, ), (1, ))
    assert_size_stride(arg160_1, (256, ), (1, ))
    assert_size_stride(arg161_1, (1024, 256), (256, 1))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (256, 1024), (1024, 1))
    assert_size_stride(arg164_1, (256, ), (1, ))
    assert_size_stride(arg165_1, (256, ), (1, ))
    assert_size_stride(arg166_1, (256, ), (1, ))
    assert_size_stride(arg167_1, (256, 256), (256, 1))
    assert_size_stride(arg168_1, (256, ), (1, ))
    assert_size_stride(arg169_1, (256, 256), (256, 1))
    assert_size_stride(arg170_1, (256, ), (1, ))
    assert_size_stride(arg171_1, (256, 256), (256, 1))
    assert_size_stride(arg172_1, (256, ), (1, ))
    assert_size_stride(arg173_1, (256, 256), (256, 1))
    assert_size_stride(arg174_1, (256, ), (1, ))
    assert_size_stride(arg175_1, (256, ), (1, ))
    assert_size_stride(arg176_1, (256, ), (1, ))
    assert_size_stride(arg177_1, (1024, 256), (256, 1))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (256, 1024), (1024, 1))
    assert_size_stride(arg180_1, (256, ), (1, ))
    assert_size_stride(arg181_1, (256, ), (1, ))
    assert_size_stride(arg182_1, (256, ), (1, ))
    assert_size_stride(arg183_1, (256, 256), (256, 1))
    assert_size_stride(arg184_1, (256, ), (1, ))
    assert_size_stride(arg185_1, (256, 256), (256, 1))
    assert_size_stride(arg186_1, (256, ), (1, ))
    assert_size_stride(arg187_1, (256, 256), (256, 1))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (256, 256), (256, 1))
    assert_size_stride(arg190_1, (256, ), (1, ))
    assert_size_stride(arg191_1, (256, ), (1, ))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (1024, 256), (256, 1))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (256, 1024), (1024, 1))
    assert_size_stride(arg196_1, (256, ), (1, ))
    assert_size_stride(arg197_1, (256, ), (1, ))
    assert_size_stride(arg198_1, (256, ), (1, ))
    assert_size_stride(arg199_1, (2, 256), (256, 1))
    assert_size_stride(arg200_1, (2, ), (1, ))
    assert_size_stride(arg201_1, (1, 512), (512, 1))
    assert_size_stride(arg202_1, (1, 512), (512, 1))
    assert_size_stride(arg203_1, (1, 512), (512, 1))
    assert_size_stride(arg204_1, (1, ), (1, ))
    assert_size_stride(arg205_1, (1, ), (1, ))
    buf0 = empty((1, 512, 128), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf4 = empty((1, 512, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_0(c_void_p(arg203_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg0_1
    del arg1_1
    del arg201_1
    del arg202_1
    del arg203_1
    del arg2_1
    del arg3_1
    del arg4_1
    del buf0
    buf5 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg6_1, reinterpret_tensor(buf4, (512, 128), (128, 1), 0), reinterpret_tensor(arg5_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf5)
    del arg5_1
    del arg6_1
    del buf4
    buf6 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf5, (512, 256), (256, 1), 0), reinterpret_tensor(arg7_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf6)
    del arg7_1
    del arg8_1
    buf7 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___electra_encoder_layer_0_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf5, (512, 256), (256, 1), 0), reinterpret_tensor(arg9_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf7)
    del arg10_1
    del arg9_1
    buf8 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___electra_encoder_layer_0_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf5, (512, 256), (256, 1), 0), reinterpret_tensor(arg11_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf8)
    del arg11_1
    del arg12_1
    buf9 = reinterpret_tensor(buf6, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf6  # reuse
    buf10 = reinterpret_tensor(buf7, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf7  # reuse
    buf11 = reinterpret_tensor(buf8, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf8  # reuse
    cpp_fused_1(c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf12 = aten._scaled_dot_product_flash_attention(buf9, buf10, buf11, scale=0.125)
    del buf10
    buf13 = buf12[0]
    del buf12
    buf20 = reinterpret_tensor(buf9, (512, 256), (256, 1), 0); del buf9  # reuse
    # Source Nodes: [hidden_states_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf13, (512, 256), (256, 1), 0), reinterpret_tensor(arg13_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf20)
    del arg13_1
    del arg14_1
    buf21 = buf2; del buf2  # reuse
    buf22 = buf1; del buf1  # reuse
    buf24 = reinterpret_tensor(buf13, (1, 512, 256), (131072, 256, 1), 0); del buf13  # reuse
    cpp_fused_add_native_layer_norm_2(c_void_p(buf20.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()))
    del arg15_1
    del arg16_1
    buf25 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf24, (512, 256), (256, 1), 0), reinterpret_tensor(arg17_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf25)
    del arg17_1
    del arg18_1
    buf26 = reinterpret_tensor(buf25, (1, 512, 1024), (524288, 1024, 1), 0); del buf25  # reuse
    cpp_fused_gelu_3(c_void_p(buf26.data_ptr()))
    buf27 = buf5; del buf5  # reuse
    # Source Nodes: [hidden_states_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf26, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg19_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf27)
    del arg19_1
    del arg20_1
    buf28 = buf22; del buf22  # reuse
    buf29 = buf21; del buf21  # reuse
    buf31 = reinterpret_tensor(buf20, (1, 512, 256), (131072, 256, 1), 0); del buf20  # reuse
    cpp_fused_add_native_layer_norm_4(c_void_p(buf27.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()))
    del arg21_1
    del arg22_1
    buf32 = buf27; del buf27  # reuse
    # Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg24_1, reinterpret_tensor(buf31, (512, 256), (256, 1), 0), reinterpret_tensor(arg23_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf32)
    del arg23_1
    del arg24_1
    buf33 = reinterpret_tensor(buf24, (512, 256), (256, 1), 0); del buf24  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_1_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg26_1, reinterpret_tensor(buf31, (512, 256), (256, 1), 0), reinterpret_tensor(arg25_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf33)
    del arg25_1
    del arg26_1
    buf34 = reinterpret_tensor(buf11, (512, 256), (256, 1), 0); del buf11  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_1_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg28_1, reinterpret_tensor(buf31, (512, 256), (256, 1), 0), reinterpret_tensor(arg27_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf34)
    del arg27_1
    del arg28_1
    buf35 = reinterpret_tensor(buf32, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf32  # reuse
    buf36 = reinterpret_tensor(buf33, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf33  # reuse
    buf37 = reinterpret_tensor(buf34, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf34  # reuse
    cpp_fused_5(c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf38 = aten._scaled_dot_product_flash_attention(buf35, buf36, buf37, scale=0.125)
    del buf35
    buf39 = buf38[0]
    del buf38
    buf46 = reinterpret_tensor(buf37, (512, 256), (256, 1), 0); del buf37  # reuse
    # Source Nodes: [hidden_states_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg30_1, reinterpret_tensor(buf39, (512, 256), (256, 1), 0), reinterpret_tensor(arg29_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf46)
    del arg29_1
    del arg30_1
    buf47 = buf29; del buf29  # reuse
    buf48 = buf28; del buf28  # reuse
    buf50 = reinterpret_tensor(buf39, (1, 512, 256), (131072, 256, 1), 0); del buf39  # reuse
    cpp_fused_add_native_layer_norm_6(c_void_p(buf46.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()))
    del arg31_1
    del arg32_1
    buf51 = reinterpret_tensor(buf26, (512, 1024), (1024, 1), 0); del buf26  # reuse
    # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg34_1, reinterpret_tensor(buf50, (512, 256), (256, 1), 0), reinterpret_tensor(arg33_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf51)
    del arg33_1
    del arg34_1
    buf52 = reinterpret_tensor(buf51, (1, 512, 1024), (524288, 1024, 1), 0); del buf51  # reuse
    cpp_fused_gelu_7(c_void_p(buf52.data_ptr()))
    buf53 = buf46; del buf46  # reuse
    # Source Nodes: [hidden_states_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg36_1, reinterpret_tensor(buf52, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg35_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf53)
    del arg35_1
    del arg36_1
    buf54 = buf48; del buf48  # reuse
    buf55 = buf47; del buf47  # reuse
    buf57 = buf31; del buf31  # reuse
    cpp_fused_add_native_layer_norm_8(c_void_p(buf53.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg37_1
    del arg38_1
    buf58 = buf53; del buf53  # reuse
    # Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg40_1, reinterpret_tensor(buf57, (512, 256), (256, 1), 0), reinterpret_tensor(arg39_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf58)
    del arg39_1
    del arg40_1
    buf59 = reinterpret_tensor(buf50, (512, 256), (256, 1), 0); del buf50  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_2_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg42_1, reinterpret_tensor(buf57, (512, 256), (256, 1), 0), reinterpret_tensor(arg41_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf59)
    del arg41_1
    del arg42_1
    buf60 = reinterpret_tensor(buf36, (512, 256), (256, 1), 0); del buf36  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_2_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg44_1, reinterpret_tensor(buf57, (512, 256), (256, 1), 0), reinterpret_tensor(arg43_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf60)
    del arg43_1
    del arg44_1
    buf61 = reinterpret_tensor(buf58, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf58  # reuse
    buf62 = reinterpret_tensor(buf59, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf59  # reuse
    buf63 = reinterpret_tensor(buf60, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf60  # reuse
    cpp_fused_9(c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf64 = aten._scaled_dot_product_flash_attention(buf61, buf62, buf63, scale=0.125)
    del buf61
    buf65 = buf64[0]
    del buf64
    buf72 = reinterpret_tensor(buf63, (512, 256), (256, 1), 0); del buf63  # reuse
    # Source Nodes: [hidden_states_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg46_1, reinterpret_tensor(buf65, (512, 256), (256, 1), 0), reinterpret_tensor(arg45_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf72)
    del arg45_1
    del arg46_1
    buf73 = buf55; del buf55  # reuse
    buf74 = buf54; del buf54  # reuse
    buf76 = reinterpret_tensor(buf65, (1, 512, 256), (131072, 256, 1), 0); del buf65  # reuse
    cpp_fused_add_native_layer_norm_10(c_void_p(buf72.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf76.data_ptr()))
    del arg47_1
    del arg48_1
    buf77 = reinterpret_tensor(buf52, (512, 1024), (1024, 1), 0); del buf52  # reuse
    # Source Nodes: [hidden_states_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg50_1, reinterpret_tensor(buf76, (512, 256), (256, 1), 0), reinterpret_tensor(arg49_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf77)
    del arg49_1
    del arg50_1
    buf78 = reinterpret_tensor(buf77, (1, 512, 1024), (524288, 1024, 1), 0); del buf77  # reuse
    cpp_fused_gelu_11(c_void_p(buf78.data_ptr()))
    buf79 = buf72; del buf72  # reuse
    # Source Nodes: [hidden_states_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg52_1, reinterpret_tensor(buf78, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg51_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf79)
    del arg51_1
    del arg52_1
    buf80 = buf74; del buf74  # reuse
    buf81 = buf73; del buf73  # reuse
    buf83 = buf57; del buf57  # reuse
    cpp_fused_add_native_layer_norm_12(c_void_p(buf79.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()))
    del arg53_1
    del arg54_1
    buf84 = buf79; del buf79  # reuse
    # Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg56_1, reinterpret_tensor(buf83, (512, 256), (256, 1), 0), reinterpret_tensor(arg55_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf84)
    del arg55_1
    del arg56_1
    buf85 = reinterpret_tensor(buf76, (512, 256), (256, 1), 0); del buf76  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_3_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg58_1, reinterpret_tensor(buf83, (512, 256), (256, 1), 0), reinterpret_tensor(arg57_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf85)
    del arg57_1
    del arg58_1
    buf86 = reinterpret_tensor(buf62, (512, 256), (256, 1), 0); del buf62  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_3_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg60_1, reinterpret_tensor(buf83, (512, 256), (256, 1), 0), reinterpret_tensor(arg59_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf86)
    del arg59_1
    del arg60_1
    buf87 = reinterpret_tensor(buf84, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf84  # reuse
    buf88 = reinterpret_tensor(buf85, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf85  # reuse
    buf89 = reinterpret_tensor(buf86, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf86  # reuse
    cpp_fused_13(c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf90 = aten._scaled_dot_product_flash_attention(buf87, buf88, buf89, scale=0.125)
    del buf87
    buf91 = buf90[0]
    del buf90
    buf98 = reinterpret_tensor(buf89, (512, 256), (256, 1), 0); del buf89  # reuse
    # Source Nodes: [hidden_states_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg62_1, reinterpret_tensor(buf91, (512, 256), (256, 1), 0), reinterpret_tensor(arg61_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf98)
    del arg61_1
    del arg62_1
    buf99 = buf81; del buf81  # reuse
    buf100 = buf80; del buf80  # reuse
    buf102 = reinterpret_tensor(buf91, (1, 512, 256), (131072, 256, 1), 0); del buf91  # reuse
    cpp_fused_add_native_layer_norm_14(c_void_p(buf98.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()))
    del arg63_1
    del arg64_1
    buf103 = reinterpret_tensor(buf78, (512, 1024), (1024, 1), 0); del buf78  # reuse
    # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg66_1, reinterpret_tensor(buf102, (512, 256), (256, 1), 0), reinterpret_tensor(arg65_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf103)
    del arg65_1
    del arg66_1
    buf104 = reinterpret_tensor(buf103, (1, 512, 1024), (524288, 1024, 1), 0); del buf103  # reuse
    cpp_fused_gelu_15(c_void_p(buf104.data_ptr()))
    buf105 = buf98; del buf98  # reuse
    # Source Nodes: [hidden_states_34], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg68_1, reinterpret_tensor(buf104, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg67_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf105)
    del arg67_1
    del arg68_1
    buf106 = buf99; del buf99  # reuse
    buf107 = buf100; del buf100  # reuse
    buf109 = buf83; del buf83  # reuse
    cpp_fused_add_native_layer_norm_16(c_void_p(buf105.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()))
    del arg69_1
    del arg70_1
    buf110 = buf105; del buf105  # reuse
    # Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg72_1, reinterpret_tensor(buf109, (512, 256), (256, 1), 0), reinterpret_tensor(arg71_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf110)
    del arg71_1
    del arg72_1
    buf111 = reinterpret_tensor(buf102, (512, 256), (256, 1), 0); del buf102  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_4_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg74_1, reinterpret_tensor(buf109, (512, 256), (256, 1), 0), reinterpret_tensor(arg73_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf111)
    del arg73_1
    del arg74_1
    buf112 = reinterpret_tensor(buf88, (512, 256), (256, 1), 0); del buf88  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_4_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg76_1, reinterpret_tensor(buf109, (512, 256), (256, 1), 0), reinterpret_tensor(arg75_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf112)
    del arg75_1
    del arg76_1
    buf113 = reinterpret_tensor(buf110, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf110  # reuse
    buf114 = reinterpret_tensor(buf111, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf111  # reuse
    buf115 = reinterpret_tensor(buf112, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf112  # reuse
    cpp_fused_17(c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf116 = aten._scaled_dot_product_flash_attention(buf113, buf114, buf115, scale=0.125)
    del buf113
    buf117 = buf116[0]
    del buf116
    buf124 = reinterpret_tensor(buf115, (512, 256), (256, 1), 0); del buf115  # reuse
    # Source Nodes: [hidden_states_38], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg78_1, reinterpret_tensor(buf117, (512, 256), (256, 1), 0), reinterpret_tensor(arg77_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf124)
    del arg77_1
    del arg78_1
    buf125 = buf107; del buf107  # reuse
    buf126 = buf106; del buf106  # reuse
    buf128 = reinterpret_tensor(buf117, (1, 512, 256), (131072, 256, 1), 0); del buf117  # reuse
    cpp_fused_add_native_layer_norm_18(c_void_p(buf124.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()))
    del arg79_1
    del arg80_1
    buf129 = reinterpret_tensor(buf104, (512, 1024), (1024, 1), 0); del buf104  # reuse
    # Source Nodes: [hidden_states_41], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg82_1, reinterpret_tensor(buf128, (512, 256), (256, 1), 0), reinterpret_tensor(arg81_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf129)
    del arg81_1
    del arg82_1
    buf130 = reinterpret_tensor(buf129, (1, 512, 1024), (524288, 1024, 1), 0); del buf129  # reuse
    cpp_fused_gelu_19(c_void_p(buf130.data_ptr()))
    buf131 = buf124; del buf124  # reuse
    # Source Nodes: [hidden_states_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg84_1, reinterpret_tensor(buf130, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg83_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf131)
    del arg83_1
    del arg84_1
    buf132 = buf126; del buf126  # reuse
    buf133 = buf125; del buf125  # reuse
    buf135 = buf109; del buf109  # reuse
    cpp_fused_add_native_layer_norm_20(c_void_p(buf131.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()))
    del arg85_1
    del arg86_1
    buf136 = buf131; del buf131  # reuse
    # Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg88_1, reinterpret_tensor(buf135, (512, 256), (256, 1), 0), reinterpret_tensor(arg87_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf136)
    del arg87_1
    del arg88_1
    buf137 = reinterpret_tensor(buf128, (512, 256), (256, 1), 0); del buf128  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_5_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg90_1, reinterpret_tensor(buf135, (512, 256), (256, 1), 0), reinterpret_tensor(arg89_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf137)
    del arg89_1
    del arg90_1
    buf138 = reinterpret_tensor(buf114, (512, 256), (256, 1), 0); del buf114  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_5_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg92_1, reinterpret_tensor(buf135, (512, 256), (256, 1), 0), reinterpret_tensor(arg91_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf138)
    del arg91_1
    del arg92_1
    buf139 = reinterpret_tensor(buf136, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf136  # reuse
    buf140 = reinterpret_tensor(buf137, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf137  # reuse
    buf141 = reinterpret_tensor(buf138, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf138  # reuse
    cpp_fused_21(c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf142 = aten._scaled_dot_product_flash_attention(buf139, buf140, buf141, scale=0.125)
    del buf139
    buf143 = buf142[0]
    del buf142
    buf150 = reinterpret_tensor(buf141, (512, 256), (256, 1), 0); del buf141  # reuse
    # Source Nodes: [hidden_states_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg94_1, reinterpret_tensor(buf143, (512, 256), (256, 1), 0), reinterpret_tensor(arg93_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf150)
    del arg93_1
    del arg94_1
    buf151 = buf133; del buf133  # reuse
    buf152 = buf132; del buf132  # reuse
    buf154 = reinterpret_tensor(buf143, (1, 512, 256), (131072, 256, 1), 0); del buf143  # reuse
    cpp_fused_add_native_layer_norm_22(c_void_p(buf150.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()))
    del arg95_1
    del arg96_1
    buf155 = reinterpret_tensor(buf130, (512, 1024), (1024, 1), 0); del buf130  # reuse
    # Source Nodes: [hidden_states_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg98_1, reinterpret_tensor(buf154, (512, 256), (256, 1), 0), reinterpret_tensor(arg97_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf155)
    del arg97_1
    del arg98_1
    buf156 = reinterpret_tensor(buf155, (1, 512, 1024), (524288, 1024, 1), 0); del buf155  # reuse
    cpp_fused_gelu_23(c_void_p(buf156.data_ptr()))
    buf157 = buf150; del buf150  # reuse
    # Source Nodes: [hidden_states_52], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg100_1, reinterpret_tensor(buf156, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg99_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf157)
    del arg100_1
    del arg99_1
    buf158 = buf152; del buf152  # reuse
    buf159 = buf151; del buf151  # reuse
    buf161 = buf135; del buf135  # reuse
    cpp_fused_add_native_layer_norm_24(c_void_p(buf157.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()))
    del arg101_1
    del arg102_1
    buf162 = buf157; del buf157  # reuse
    # Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg104_1, reinterpret_tensor(buf161, (512, 256), (256, 1), 0), reinterpret_tensor(arg103_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf162)
    del arg103_1
    del arg104_1
    buf163 = reinterpret_tensor(buf154, (512, 256), (256, 1), 0); del buf154  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_6_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg106_1, reinterpret_tensor(buf161, (512, 256), (256, 1), 0), reinterpret_tensor(arg105_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf163)
    del arg105_1
    del arg106_1
    buf164 = reinterpret_tensor(buf140, (512, 256), (256, 1), 0); del buf140  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_6_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg108_1, reinterpret_tensor(buf161, (512, 256), (256, 1), 0), reinterpret_tensor(arg107_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf164)
    del arg107_1
    del arg108_1
    buf165 = reinterpret_tensor(buf162, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf162  # reuse
    buf166 = reinterpret_tensor(buf163, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf163  # reuse
    buf167 = reinterpret_tensor(buf164, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf164  # reuse
    cpp_fused_25(c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf168 = aten._scaled_dot_product_flash_attention(buf165, buf166, buf167, scale=0.125)
    del buf165
    buf169 = buf168[0]
    del buf168
    buf176 = reinterpret_tensor(buf167, (512, 256), (256, 1), 0); del buf167  # reuse
    # Source Nodes: [hidden_states_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg110_1, reinterpret_tensor(buf169, (512, 256), (256, 1), 0), reinterpret_tensor(arg109_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf176)
    del arg109_1
    del arg110_1
    buf177 = buf159; del buf159  # reuse
    buf178 = buf158; del buf158  # reuse
    buf180 = reinterpret_tensor(buf169, (1, 512, 256), (131072, 256, 1), 0); del buf169  # reuse
    cpp_fused_add_native_layer_norm_26(c_void_p(buf176.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()))
    del arg111_1
    del arg112_1
    buf181 = reinterpret_tensor(buf156, (512, 1024), (1024, 1), 0); del buf156  # reuse
    # Source Nodes: [hidden_states_59], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg114_1, reinterpret_tensor(buf180, (512, 256), (256, 1), 0), reinterpret_tensor(arg113_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf181)
    del arg113_1
    del arg114_1
    buf182 = reinterpret_tensor(buf181, (1, 512, 1024), (524288, 1024, 1), 0); del buf181  # reuse
    cpp_fused_gelu_27(c_void_p(buf182.data_ptr()))
    buf183 = buf176; del buf176  # reuse
    # Source Nodes: [hidden_states_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg116_1, reinterpret_tensor(buf182, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg115_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf183)
    del arg115_1
    del arg116_1
    buf184 = buf178; del buf178  # reuse
    buf185 = buf177; del buf177  # reuse
    buf187 = buf161; del buf161  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf183.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()))
    del arg117_1
    del arg118_1
    buf188 = buf183; del buf183  # reuse
    # Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg120_1, reinterpret_tensor(buf187, (512, 256), (256, 1), 0), reinterpret_tensor(arg119_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf188)
    del arg119_1
    del arg120_1
    buf189 = reinterpret_tensor(buf180, (512, 256), (256, 1), 0); del buf180  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_7_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg122_1, reinterpret_tensor(buf187, (512, 256), (256, 1), 0), reinterpret_tensor(arg121_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf189)
    del arg121_1
    del arg122_1
    buf190 = reinterpret_tensor(buf166, (512, 256), (256, 1), 0); del buf166  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_7_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg124_1, reinterpret_tensor(buf187, (512, 256), (256, 1), 0), reinterpret_tensor(arg123_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf190)
    del arg123_1
    del arg124_1
    buf191 = reinterpret_tensor(buf188, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf188  # reuse
    buf192 = reinterpret_tensor(buf189, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf189  # reuse
    buf193 = reinterpret_tensor(buf190, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf190  # reuse
    cpp_fused_29(c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf194 = aten._scaled_dot_product_flash_attention(buf191, buf192, buf193, scale=0.125)
    del buf191
    buf195 = buf194[0]
    del buf194
    buf202 = reinterpret_tensor(buf193, (512, 256), (256, 1), 0); del buf193  # reuse
    # Source Nodes: [hidden_states_65], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg126_1, reinterpret_tensor(buf195, (512, 256), (256, 1), 0), reinterpret_tensor(arg125_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf202)
    del arg125_1
    del arg126_1
    buf203 = buf185; del buf185  # reuse
    buf204 = buf184; del buf184  # reuse
    buf206 = reinterpret_tensor(buf195, (1, 512, 256), (131072, 256, 1), 0); del buf195  # reuse
    cpp_fused_add_native_layer_norm_30(c_void_p(buf202.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf206.data_ptr()))
    del arg127_1
    del arg128_1
    buf207 = reinterpret_tensor(buf182, (512, 1024), (1024, 1), 0); del buf182  # reuse
    # Source Nodes: [hidden_states_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg130_1, reinterpret_tensor(buf206, (512, 256), (256, 1), 0), reinterpret_tensor(arg129_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf207)
    del arg129_1
    del arg130_1
    buf208 = reinterpret_tensor(buf207, (1, 512, 1024), (524288, 1024, 1), 0); del buf207  # reuse
    cpp_fused_gelu_31(c_void_p(buf208.data_ptr()))
    buf209 = buf202; del buf202  # reuse
    # Source Nodes: [hidden_states_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg132_1, reinterpret_tensor(buf208, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg131_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf209)
    del arg131_1
    del arg132_1
    buf210 = buf204; del buf204  # reuse
    buf211 = buf203; del buf203  # reuse
    buf213 = buf187; del buf187  # reuse
    cpp_fused_add_native_layer_norm_32(c_void_p(buf209.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf213.data_ptr()))
    del arg133_1
    del arg134_1
    buf214 = buf209; del buf209  # reuse
    # Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg136_1, reinterpret_tensor(buf213, (512, 256), (256, 1), 0), reinterpret_tensor(arg135_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf214)
    del arg135_1
    del arg136_1
    buf215 = reinterpret_tensor(buf206, (512, 256), (256, 1), 0); del buf206  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_8_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf213, (512, 256), (256, 1), 0), reinterpret_tensor(arg137_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf215)
    del arg137_1
    del arg138_1
    buf216 = reinterpret_tensor(buf192, (512, 256), (256, 1), 0); del buf192  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_8_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg140_1, reinterpret_tensor(buf213, (512, 256), (256, 1), 0), reinterpret_tensor(arg139_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf216)
    del arg139_1
    del arg140_1
    buf217 = reinterpret_tensor(buf214, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf214  # reuse
    buf218 = reinterpret_tensor(buf215, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf215  # reuse
    buf219 = reinterpret_tensor(buf216, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf216  # reuse
    cpp_fused_33(c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf220 = aten._scaled_dot_product_flash_attention(buf217, buf218, buf219, scale=0.125)
    del buf217
    buf221 = buf220[0]
    del buf220
    buf228 = reinterpret_tensor(buf219, (512, 256), (256, 1), 0); del buf219  # reuse
    # Source Nodes: [hidden_states_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg142_1, reinterpret_tensor(buf221, (512, 256), (256, 1), 0), reinterpret_tensor(arg141_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf228)
    del arg141_1
    del arg142_1
    buf229 = buf211; del buf211  # reuse
    buf230 = buf210; del buf210  # reuse
    buf232 = reinterpret_tensor(buf221, (1, 512, 256), (131072, 256, 1), 0); del buf221  # reuse
    cpp_fused_add_native_layer_norm_34(c_void_p(buf228.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()))
    del arg143_1
    del arg144_1
    buf233 = reinterpret_tensor(buf208, (512, 1024), (1024, 1), 0); del buf208  # reuse
    # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg146_1, reinterpret_tensor(buf232, (512, 256), (256, 1), 0), reinterpret_tensor(arg145_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf233)
    del arg145_1
    del arg146_1
    buf234 = reinterpret_tensor(buf233, (1, 512, 1024), (524288, 1024, 1), 0); del buf233  # reuse
    cpp_fused_gelu_35(c_void_p(buf234.data_ptr()))
    buf235 = buf228; del buf228  # reuse
    # Source Nodes: [hidden_states_79], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg148_1, reinterpret_tensor(buf234, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg147_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf235)
    del arg147_1
    del arg148_1
    buf236 = buf230; del buf230  # reuse
    buf237 = buf229; del buf229  # reuse
    buf239 = buf213; del buf213  # reuse
    cpp_fused_add_native_layer_norm_36(c_void_p(buf235.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()))
    del arg149_1
    del arg150_1
    buf240 = buf235; del buf235  # reuse
    # Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg152_1, reinterpret_tensor(buf239, (512, 256), (256, 1), 0), reinterpret_tensor(arg151_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf240)
    del arg151_1
    del arg152_1
    buf241 = reinterpret_tensor(buf232, (512, 256), (256, 1), 0); del buf232  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_9_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg154_1, reinterpret_tensor(buf239, (512, 256), (256, 1), 0), reinterpret_tensor(arg153_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf241)
    del arg153_1
    del arg154_1
    buf242 = reinterpret_tensor(buf218, (512, 256), (256, 1), 0); del buf218  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_9_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg156_1, reinterpret_tensor(buf239, (512, 256), (256, 1), 0), reinterpret_tensor(arg155_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf242)
    del arg155_1
    del arg156_1
    buf243 = reinterpret_tensor(buf240, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf240  # reuse
    buf244 = reinterpret_tensor(buf241, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf241  # reuse
    buf245 = reinterpret_tensor(buf242, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf242  # reuse
    cpp_fused_37(c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf246 = aten._scaled_dot_product_flash_attention(buf243, buf244, buf245, scale=0.125)
    del buf243
    buf247 = buf246[0]
    del buf246
    buf254 = reinterpret_tensor(buf245, (512, 256), (256, 1), 0); del buf245  # reuse
    # Source Nodes: [hidden_states_83], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg158_1, reinterpret_tensor(buf247, (512, 256), (256, 1), 0), reinterpret_tensor(arg157_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf254)
    del arg157_1
    del arg158_1
    buf255 = buf237; del buf237  # reuse
    buf256 = buf236; del buf236  # reuse
    buf258 = reinterpret_tensor(buf247, (1, 512, 256), (131072, 256, 1), 0); del buf247  # reuse
    cpp_fused_add_native_layer_norm_38(c_void_p(buf254.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()))
    del arg159_1
    del arg160_1
    buf259 = reinterpret_tensor(buf234, (512, 1024), (1024, 1), 0); del buf234  # reuse
    # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg162_1, reinterpret_tensor(buf258, (512, 256), (256, 1), 0), reinterpret_tensor(arg161_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf259)
    del arg161_1
    del arg162_1
    buf260 = reinterpret_tensor(buf259, (1, 512, 1024), (524288, 1024, 1), 0); del buf259  # reuse
    cpp_fused_gelu_39(c_void_p(buf260.data_ptr()))
    buf261 = buf254; del buf254  # reuse
    # Source Nodes: [hidden_states_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg164_1, reinterpret_tensor(buf260, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg163_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf261)
    del arg163_1
    del arg164_1
    buf262 = buf256; del buf256  # reuse
    buf263 = buf255; del buf255  # reuse
    buf265 = buf239; del buf239  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf261.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf265.data_ptr()))
    del arg165_1
    del arg166_1
    buf266 = buf261; del buf261  # reuse
    # Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg168_1, reinterpret_tensor(buf265, (512, 256), (256, 1), 0), reinterpret_tensor(arg167_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf266)
    del arg167_1
    del arg168_1
    buf267 = reinterpret_tensor(buf258, (512, 256), (256, 1), 0); del buf258  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_10_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg170_1, reinterpret_tensor(buf265, (512, 256), (256, 1), 0), reinterpret_tensor(arg169_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf267)
    del arg169_1
    del arg170_1
    buf268 = reinterpret_tensor(buf244, (512, 256), (256, 1), 0); del buf244  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_10_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg172_1, reinterpret_tensor(buf265, (512, 256), (256, 1), 0), reinterpret_tensor(arg171_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf268)
    del arg171_1
    del arg172_1
    buf269 = reinterpret_tensor(buf266, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf266  # reuse
    buf270 = reinterpret_tensor(buf267, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf267  # reuse
    buf271 = reinterpret_tensor(buf268, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf268  # reuse
    cpp_fused_41(c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf272 = aten._scaled_dot_product_flash_attention(buf269, buf270, buf271, scale=0.125)
    del buf269
    buf273 = buf272[0]
    del buf272
    buf280 = reinterpret_tensor(buf271, (512, 256), (256, 1), 0); del buf271  # reuse
    # Source Nodes: [hidden_states_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg174_1, reinterpret_tensor(buf273, (512, 256), (256, 1), 0), reinterpret_tensor(arg173_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf280)
    del arg173_1
    del arg174_1
    buf281 = buf263; del buf263  # reuse
    buf282 = buf262; del buf262  # reuse
    buf284 = reinterpret_tensor(buf273, (1, 512, 256), (131072, 256, 1), 0); del buf273  # reuse
    cpp_fused_add_native_layer_norm_42(c_void_p(buf280.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()))
    del arg175_1
    del arg176_1
    buf285 = reinterpret_tensor(buf260, (512, 1024), (1024, 1), 0); del buf260  # reuse
    # Source Nodes: [hidden_states_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg178_1, reinterpret_tensor(buf284, (512, 256), (256, 1), 0), reinterpret_tensor(arg177_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf285)
    del arg177_1
    del arg178_1
    buf286 = reinterpret_tensor(buf285, (1, 512, 1024), (524288, 1024, 1), 0); del buf285  # reuse
    cpp_fused_gelu_43(c_void_p(buf286.data_ptr()))
    buf287 = buf280; del buf280  # reuse
    # Source Nodes: [hidden_states_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg180_1, reinterpret_tensor(buf286, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg179_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf287)
    del arg179_1
    del arg180_1
    buf288 = buf282; del buf282  # reuse
    buf289 = buf281; del buf281  # reuse
    buf291 = buf265; del buf265  # reuse
    cpp_fused_add_native_layer_norm_44(c_void_p(buf287.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()))
    del arg181_1
    del arg182_1
    buf292 = buf287; del buf287  # reuse
    # Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg184_1, reinterpret_tensor(buf291, (512, 256), (256, 1), 0), reinterpret_tensor(arg183_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf292)
    del arg183_1
    del arg184_1
    buf293 = reinterpret_tensor(buf284, (512, 256), (256, 1), 0); del buf284  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_11_attention_self_key], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg186_1, reinterpret_tensor(buf291, (512, 256), (256, 1), 0), reinterpret_tensor(arg185_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf293)
    del arg185_1
    del arg186_1
    buf294 = reinterpret_tensor(buf270, (512, 256), (256, 1), 0); del buf270  # reuse
    # Source Nodes: [l__mod___electra_encoder_layer_11_attention_self_value], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg188_1, reinterpret_tensor(buf291, (512, 256), (256, 1), 0), reinterpret_tensor(arg187_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf294)
    del arg187_1
    del arg188_1
    buf295 = reinterpret_tensor(buf292, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf292  # reuse
    buf296 = reinterpret_tensor(buf293, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf293  # reuse
    buf297 = reinterpret_tensor(buf294, (1, 4, 512, 64), (131072, 64, 256, 1), 0); del buf294  # reuse
    cpp_fused_45(c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf298 = aten._scaled_dot_product_flash_attention(buf295, buf296, buf297, scale=0.125)
    del buf295
    del buf296
    buf299 = buf298[0]
    del buf298
    buf306 = reinterpret_tensor(buf297, (512, 256), (256, 1), 0); del buf297  # reuse
    # Source Nodes: [hidden_states_101], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg190_1, reinterpret_tensor(buf299, (512, 256), (256, 1), 0), reinterpret_tensor(arg189_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf306)
    del arg189_1
    del arg190_1
    buf307 = buf289; del buf289  # reuse
    buf308 = buf288; del buf288  # reuse
    buf310 = reinterpret_tensor(buf299, (1, 512, 256), (131072, 256, 1), 0); del buf299  # reuse
    cpp_fused_add_native_layer_norm_46(c_void_p(buf306.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf310.data_ptr()))
    del arg191_1
    del arg192_1
    buf311 = reinterpret_tensor(buf286, (512, 1024), (1024, 1), 0); del buf286  # reuse
    # Source Nodes: [hidden_states_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg194_1, reinterpret_tensor(buf310, (512, 256), (256, 1), 0), reinterpret_tensor(arg193_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf311)
    del arg193_1
    del arg194_1
    buf312 = reinterpret_tensor(buf311, (1, 512, 1024), (524288, 1024, 1), 0); del buf311  # reuse
    cpp_fused_gelu_47(c_void_p(buf312.data_ptr()))
    buf313 = buf306; del buf306  # reuse
    # Source Nodes: [hidden_states_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg196_1, reinterpret_tensor(buf312, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg195_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf313)
    del arg195_1
    del arg196_1
    del buf312
    buf314 = buf308; del buf308  # reuse
    buf315 = buf307; del buf307  # reuse
    buf317 = buf291; del buf291  # reuse
    cpp_fused_add_native_layer_norm_48(c_void_p(buf313.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf317.data_ptr()))
    del arg197_1
    del arg198_1
    del buf310
    del buf313
    buf318 = empty((512, 2), device='cpu', dtype=torch.float32)
    # Source Nodes: [logits], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg200_1, reinterpret_tensor(buf317, (512, 256), (256, 1), 0), reinterpret_tensor(arg199_1, (256, 2), (1, 256), 0), alpha=1, beta=1, out=buf318)
    del arg199_1
    del arg200_1
    del buf317
    buf319 = reinterpret_tensor(buf315, (1, 512), (512, 1), 0); del buf315  # reuse
    buf320 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf321 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf322 = reinterpret_tensor(buf314, (1, 512), (512, 1), 0); del buf314  # reuse
    buf323 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf324 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf325 = reinterpret_tensor(buf320, (), (), 0); del buf320  # reuse
    buf326 = buf325; del buf325  # reuse
    cpp_fused__log_softmax_add_clamp_clone_div_nll_loss_forward_49(c_void_p(buf326.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()))
    del arg204_1
    del arg205_1
    return (buf326, buf319, buf322, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30522, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((2, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((2, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg202_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg203_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg204_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    arg205_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ElectraForQuestionAnswering', benchmark_compiled_module)
