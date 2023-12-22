
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


cpp_fused_add_arange_embedding_mul_native_layer_norm_0 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused__softmax_clone_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_arange_embedding_mul_native_layer_norm_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 50265);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 50265L), "index out of bounds: 0 <= tmp3 < 50265L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*tmp3)));
                        auto tmp5 = static_cast<float>(1.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = out_ptr1[static_cast<long>(x0)];
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 50265);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 50265L), "index out of bounds: 0 <= tmp3 < 50265L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*tmp3)));
                    auto tmp5 = static_cast<float>(1.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = tmp9 + tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp16 = static_cast<float>(1024.0);
                    auto tmp17 = tmp15 / tmp16;
                    auto tmp18 = static_cast<float>(1e-05);
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


cpp_fused_gelu_6 = async_compile.cpp('''
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


cpp_fused_add_arange_embedding_mul_native_layer_norm_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 50265);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 50265L), "index out of bounds: 0 <= tmp3 < 50265L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*tmp3)));
                        auto tmp5 = static_cast<float>(1.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_8 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused__softmax_clone_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_13 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_15 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused__softmax_clone_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_20 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_22 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused__softmax_clone_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_29 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused__softmax_clone_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_34 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_36 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused__softmax_clone_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_41 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_48 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_50 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused__softmax_clone_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_55 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_57 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused__softmax_clone_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_62 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_64 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused__softmax_clone_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_69 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused__softmax_clone_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_76 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_78 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused__softmax_clone_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1024L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused__log_softmax_nll_loss_forward_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (50265L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (50265L*x0))];
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
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                        auto tmp8 = in_ptr0[static_cast<long>(tmp7 + (50265L*x0))];
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg1_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg2_1, (1024, ), (1, ))
    assert_size_stride(arg3_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg5_1, (1024, ), (1, ))
    assert_size_stride(arg6_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg7_1, (1024, ), (1, ))
    assert_size_stride(arg8_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg9_1, (1024, ), (1, ))
    assert_size_stride(arg10_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg11_1, (1024, ), (1, ))
    assert_size_stride(arg12_1, (1024, ), (1, ))
    assert_size_stride(arg13_1, (1024, ), (1, ))
    assert_size_stride(arg14_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg15_1, (4096, ), (1, ))
    assert_size_stride(arg16_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg17_1, (1024, ), (1, ))
    assert_size_stride(arg18_1, (1024, ), (1, ))
    assert_size_stride(arg19_1, (1024, ), (1, ))
    assert_size_stride(arg20_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg21_1, (1024, ), (1, ))
    assert_size_stride(arg22_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg23_1, (1024, ), (1, ))
    assert_size_stride(arg24_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg25_1, (1024, ), (1, ))
    assert_size_stride(arg26_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg27_1, (1024, ), (1, ))
    assert_size_stride(arg28_1, (1024, ), (1, ))
    assert_size_stride(arg29_1, (1024, ), (1, ))
    assert_size_stride(arg30_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg31_1, (4096, ), (1, ))
    assert_size_stride(arg32_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg33_1, (1024, ), (1, ))
    assert_size_stride(arg34_1, (1024, ), (1, ))
    assert_size_stride(arg35_1, (1024, ), (1, ))
    assert_size_stride(arg36_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg39_1, (1024, ), (1, ))
    assert_size_stride(arg40_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg41_1, (1024, ), (1, ))
    assert_size_stride(arg42_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg43_1, (1024, ), (1, ))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (1024, ), (1, ))
    assert_size_stride(arg46_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg47_1, (4096, ), (1, ))
    assert_size_stride(arg48_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg49_1, (1024, ), (1, ))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (1024, ), (1, ))
    assert_size_stride(arg52_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg53_1, (1024, ), (1, ))
    assert_size_stride(arg54_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg55_1, (1024, ), (1, ))
    assert_size_stride(arg56_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg57_1, (1024, ), (1, ))
    assert_size_stride(arg58_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (1024, ), (1, ))
    assert_size_stride(arg62_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg63_1, (4096, ), (1, ))
    assert_size_stride(arg64_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg65_1, (1024, ), (1, ))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg69_1, (1024, ), (1, ))
    assert_size_stride(arg70_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg71_1, (1024, ), (1, ))
    assert_size_stride(arg72_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg73_1, (1024, ), (1, ))
    assert_size_stride(arg74_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg75_1, (1024, ), (1, ))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg79_1, (4096, ), (1, ))
    assert_size_stride(arg80_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg81_1, (1024, ), (1, ))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg85_1, (1024, ), (1, ))
    assert_size_stride(arg86_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg87_1, (1024, ), (1, ))
    assert_size_stride(arg88_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg89_1, (1024, ), (1, ))
    assert_size_stride(arg90_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (1024, ), (1, ))
    assert_size_stride(arg94_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg95_1, (4096, ), (1, ))
    assert_size_stride(arg96_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg97_1, (1024, ), (1, ))
    assert_size_stride(arg98_1, (1024, ), (1, ))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg103_1, (1024, ), (1, ))
    assert_size_stride(arg104_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg105_1, (1024, ), (1, ))
    assert_size_stride(arg106_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg107_1, (1024, ), (1, ))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg111_1, (4096, ), (1, ))
    assert_size_stride(arg112_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg113_1, (1024, ), (1, ))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg119_1, (1024, ), (1, ))
    assert_size_stride(arg120_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg121_1, (1024, ), (1, ))
    assert_size_stride(arg122_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg123_1, (1024, ), (1, ))
    assert_size_stride(arg124_1, (1024, ), (1, ))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg127_1, (4096, ), (1, ))
    assert_size_stride(arg128_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg129_1, (1024, ), (1, ))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (1024, ), (1, ))
    assert_size_stride(arg142_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg143_1, (4096, ), (1, ))
    assert_size_stride(arg144_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg151_1, (1024, ), (1, ))
    assert_size_stride(arg152_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg153_1, (1024, ), (1, ))
    assert_size_stride(arg154_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (1024, ), (1, ))
    assert_size_stride(arg158_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg159_1, (4096, ), (1, ))
    assert_size_stride(arg160_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg165_1, (1024, ), (1, ))
    assert_size_stride(arg166_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg171_1, (1024, ), (1, ))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (1024, ), (1, ))
    assert_size_stride(arg174_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg175_1, (4096, ), (1, ))
    assert_size_stride(arg176_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg177_1, (1024, ), (1, ))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (1024, ), (1, ))
    assert_size_stride(arg180_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg183_1, (1024, ), (1, ))
    assert_size_stride(arg184_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg185_1, (1024, ), (1, ))
    assert_size_stride(arg186_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg187_1, (1024, ), (1, ))
    assert_size_stride(arg188_1, (1024, ), (1, ))
    assert_size_stride(arg189_1, (1024, ), (1, ))
    assert_size_stride(arg190_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg191_1, (4096, ), (1, ))
    assert_size_stride(arg192_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg193_1, (1024, ), (1, ))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (1024, ), (1, ))
    assert_size_stride(arg196_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg197_1, (1, 128), (128, 1))
    assert_size_stride(arg198_1, (1, 128), (128, 1))
    buf0 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_arange_embedding_mul_native_layer_norm_0(c_void_p(arg197_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg2_1
    del arg3_1
    buf4 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg5_1, reinterpret_tensor(buf3, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg4_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf4)
    del arg4_1
    del arg5_1
    buf5 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg7_1, reinterpret_tensor(buf3, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg6_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf5)
    del arg6_1
    del arg7_1
    buf6 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf7 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_1(c_void_p(buf5.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    buf8 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf6, (16, 64, 128), (8192, 1, 64), 0), out=buf8)
    buf9 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf10 = buf8; del buf8  # reuse
    buf11 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_2(c_void_p(buf10.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf11.data_ptr()))
    buf12 = reinterpret_tensor(buf7, (128, 1024), (1024, 1), 0); del buf7  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg9_1, reinterpret_tensor(buf3, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg8_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf12)
    del arg8_1
    del arg9_1
    buf13 = reinterpret_tensor(buf3, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf3  # reuse
    buf14 = buf10; del buf10  # reuse
    cpp_fused__softmax_clone_3(c_void_p(buf14.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()))
    buf15 = reinterpret_tensor(buf12, (16, 128, 64), (8192, 64, 1), 0); del buf12  # reuse
    # Source Nodes: [attn_output, attn_weights_3], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf14, reinterpret_tensor(buf13, (16, 128, 64), (8192, 64, 1), 0), out=buf15)
    buf16 = reinterpret_tensor(buf5, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf5  # reuse
    cpp_fused_clone_4(c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()))
    buf17 = reinterpret_tensor(buf15, (128, 1024), (1024, 1), 0); del buf15  # reuse
    # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg11_1, reinterpret_tensor(buf16, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg10_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf17)
    del arg10_1
    del arg11_1
    buf18 = buf1; del buf1  # reuse
    buf19 = buf0; del buf0  # reuse
    buf21 = reinterpret_tensor(buf16, (1, 128, 1024), (131072, 1024, 1), 0); del buf16  # reuse
    cpp_fused_add_arange_embedding_mul_native_layer_norm_5(c_void_p(arg197_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()))
    del arg12_1
    del arg13_1
    buf22 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_0_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg15_1, reinterpret_tensor(buf21, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg14_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf22)
    del arg14_1
    del arg15_1
    buf23 = reinterpret_tensor(buf22, (1, 128, 4096), (524288, 4096, 1), 0); del buf22  # reuse
    cpp_fused_gelu_6(c_void_p(buf23.data_ptr()))
    buf24 = reinterpret_tensor(buf21, (128, 1024), (1024, 1), 0); del buf21  # reuse
    # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg17_1, reinterpret_tensor(buf23, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf24)
    del arg16_1
    del arg17_1
    buf25 = reinterpret_tensor(buf24, (1, 128, 1024), (131072, 1024, 1), 0); del buf24  # reuse
    buf26 = buf19; del buf19  # reuse
    buf27 = buf18; del buf18  # reuse
    buf29 = reinterpret_tensor(buf4, (1, 128, 1024), (131072, 1024, 1), 0); del buf4  # reuse
    cpp_fused_add_arange_embedding_mul_native_layer_norm_7(c_void_p(buf25.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf29.data_ptr()))
    del arg0_1
    del arg18_1
    del arg197_1
    del arg19_1
    del arg1_1
    buf30 = buf17; del buf17  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg21_1, reinterpret_tensor(buf29, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg20_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf30)
    del arg20_1
    del arg21_1
    buf31 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg23_1, reinterpret_tensor(buf29, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg22_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf31)
    del arg22_1
    del arg23_1
    buf32 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf33 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_8(c_void_p(buf31.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()))
    buf34 = buf14; del buf14  # reuse
    # Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf33, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf32, (16, 64, 128), (8192, 1, 64), 0), out=buf34)
    buf35 = buf11; del buf11  # reuse
    buf36 = buf34; del buf34  # reuse
    buf37 = buf9; del buf9  # reuse
    cpp_fused__softmax_9(c_void_p(buf36.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf37.data_ptr()))
    buf38 = reinterpret_tensor(buf33, (128, 1024), (1024, 1), 0); del buf33  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg25_1, reinterpret_tensor(buf29, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg24_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf38)
    del arg24_1
    del arg25_1
    buf39 = reinterpret_tensor(buf29, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf29  # reuse
    buf40 = buf36; del buf36  # reuse
    cpp_fused__softmax_clone_10(c_void_p(buf40.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()))
    buf41 = reinterpret_tensor(buf38, (16, 128, 64), (8192, 64, 1), 0); del buf38  # reuse
    # Source Nodes: [attn_output_5, attn_weights_7], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf40, reinterpret_tensor(buf39, (16, 128, 64), (8192, 64, 1), 0), out=buf41)
    buf42 = reinterpret_tensor(buf31, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf31  # reuse
    cpp_fused_clone_11(c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()))
    buf43 = reinterpret_tensor(buf41, (128, 1024), (1024, 1), 0); del buf41  # reuse
    # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg27_1, reinterpret_tensor(buf42, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg26_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf43)
    del arg26_1
    del arg27_1
    buf44 = buf27; del buf27  # reuse
    buf45 = buf26; del buf26  # reuse
    buf47 = reinterpret_tensor(buf42, (1, 128, 1024), (131072, 1024, 1), 0); del buf42  # reuse
    cpp_fused_add_native_layer_norm_12(c_void_p(buf25.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()))
    del arg28_1
    del arg29_1
    buf48 = reinterpret_tensor(buf23, (128, 4096), (4096, 1), 0); del buf23  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg31_1, reinterpret_tensor(buf47, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg30_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf48)
    del arg30_1
    del arg31_1
    buf49 = reinterpret_tensor(buf48, (1, 128, 4096), (524288, 4096, 1), 0); del buf48  # reuse
    cpp_fused_gelu_13(c_void_p(buf49.data_ptr()))
    buf50 = reinterpret_tensor(buf47, (128, 1024), (1024, 1), 0); del buf47  # reuse
    # Source Nodes: [hidden_states_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg33_1, reinterpret_tensor(buf49, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg32_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf50)
    del arg32_1
    del arg33_1
    buf51 = buf45; del buf45  # reuse
    buf52 = buf44; del buf44  # reuse
    buf54 = reinterpret_tensor(buf30, (1, 128, 1024), (131072, 1024, 1), 0); del buf30  # reuse
    cpp_fused_add_native_layer_norm_14(c_void_p(buf25.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf54.data_ptr()))
    del arg34_1
    del arg35_1
    buf55 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg37_1, reinterpret_tensor(buf54, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg36_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf55)
    del arg36_1
    del arg37_1
    buf56 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg39_1, reinterpret_tensor(buf54, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg38_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf56)
    del arg38_1
    del arg39_1
    buf57 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf58 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_15(c_void_p(buf56.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    buf59 = buf40; del buf40  # reuse
    # Source Nodes: [attn_weights_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf58, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf57, (16, 64, 128), (8192, 1, 64), 0), out=buf59)
    buf60 = buf37; del buf37  # reuse
    buf61 = buf59; del buf59  # reuse
    buf62 = buf35; del buf35  # reuse
    cpp_fused__softmax_16(c_void_p(buf61.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf62.data_ptr()))
    buf63 = reinterpret_tensor(buf58, (128, 1024), (1024, 1), 0); del buf58  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg41_1, reinterpret_tensor(buf54, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg40_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf63)
    del arg40_1
    del arg41_1
    buf64 = reinterpret_tensor(buf54, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf54  # reuse
    buf65 = buf61; del buf61  # reuse
    cpp_fused__softmax_clone_17(c_void_p(buf65.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()))
    buf66 = reinterpret_tensor(buf63, (16, 128, 64), (8192, 64, 1), 0); del buf63  # reuse
    # Source Nodes: [attn_output_10, attn_weights_11], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf65, reinterpret_tensor(buf64, (16, 128, 64), (8192, 64, 1), 0), out=buf66)
    buf67 = reinterpret_tensor(buf56, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf56  # reuse
    cpp_fused_clone_18(c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    buf68 = reinterpret_tensor(buf66, (128, 1024), (1024, 1), 0); del buf66  # reuse
    # Source Nodes: [hidden_states_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg43_1, reinterpret_tensor(buf67, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg42_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf68)
    del arg42_1
    del arg43_1
    buf69 = buf52; del buf52  # reuse
    buf70 = buf51; del buf51  # reuse
    buf72 = reinterpret_tensor(buf67, (1, 128, 1024), (131072, 1024, 1), 0); del buf67  # reuse
    cpp_fused_add_native_layer_norm_19(c_void_p(buf25.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()))
    del arg44_1
    del arg45_1
    buf73 = reinterpret_tensor(buf49, (128, 4096), (4096, 1), 0); del buf49  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg47_1, reinterpret_tensor(buf72, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg46_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf73)
    del arg46_1
    del arg47_1
    buf74 = reinterpret_tensor(buf73, (1, 128, 4096), (524288, 4096, 1), 0); del buf73  # reuse
    cpp_fused_gelu_20(c_void_p(buf74.data_ptr()))
    buf75 = reinterpret_tensor(buf72, (128, 1024), (1024, 1), 0); del buf72  # reuse
    # Source Nodes: [hidden_states_31], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg49_1, reinterpret_tensor(buf74, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg48_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf75)
    del arg48_1
    del arg49_1
    buf76 = reinterpret_tensor(buf75, (1, 128, 1024), (131072, 1024, 1), 0); del buf75  # reuse
    buf77 = buf70; del buf70  # reuse
    buf78 = buf69; del buf69  # reuse
    buf80 = reinterpret_tensor(buf55, (1, 128, 1024), (131072, 1024, 1), 0); del buf55  # reuse
    cpp_fused_add_native_layer_norm_21(c_void_p(buf76.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf80.data_ptr()))
    del arg50_1
    del arg51_1
    buf81 = buf68; del buf68  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg53_1, reinterpret_tensor(buf80, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg52_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf81)
    del arg52_1
    del arg53_1
    buf82 = buf50; del buf50  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg55_1, reinterpret_tensor(buf80, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg54_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf82)
    del arg54_1
    del arg55_1
    buf83 = reinterpret_tensor(buf43, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf43  # reuse
    buf84 = reinterpret_tensor(buf25, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf25  # reuse
    cpp_fused_clone_22(c_void_p(buf82.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    buf85 = buf65; del buf65  # reuse
    # Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf84, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf83, (16, 64, 128), (8192, 1, 64), 0), out=buf85)
    buf86 = buf62; del buf62  # reuse
    buf87 = buf85; del buf85  # reuse
    buf88 = buf60; del buf60  # reuse
    cpp_fused__softmax_23(c_void_p(buf87.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()))
    buf89 = reinterpret_tensor(buf84, (128, 1024), (1024, 1), 0); del buf84  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg57_1, reinterpret_tensor(buf80, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg56_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf89)
    del arg56_1
    del arg57_1
    buf90 = reinterpret_tensor(buf80, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf80  # reuse
    buf91 = buf87; del buf87  # reuse
    cpp_fused__softmax_clone_24(c_void_p(buf91.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()))
    buf92 = reinterpret_tensor(buf89, (16, 128, 64), (8192, 64, 1), 0); del buf89  # reuse
    # Source Nodes: [attn_output_15, attn_weights_15], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf91, reinterpret_tensor(buf90, (16, 128, 64), (8192, 64, 1), 0), out=buf92)
    buf93 = reinterpret_tensor(buf82, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf82  # reuse
    cpp_fused_clone_25(c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()))
    buf94 = reinterpret_tensor(buf92, (128, 1024), (1024, 1), 0); del buf92  # reuse
    # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg59_1, reinterpret_tensor(buf93, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg58_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf94)
    del arg58_1
    del arg59_1
    buf95 = buf78; del buf78  # reuse
    buf96 = buf77; del buf77  # reuse
    buf98 = reinterpret_tensor(buf93, (1, 128, 1024), (131072, 1024, 1), 0); del buf93  # reuse
    cpp_fused_add_native_layer_norm_26(c_void_p(buf76.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf98.data_ptr()))
    del arg60_1
    del arg61_1
    buf99 = reinterpret_tensor(buf74, (128, 4096), (4096, 1), 0); del buf74  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg63_1, reinterpret_tensor(buf98, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg62_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf99)
    del arg62_1
    del arg63_1
    buf100 = reinterpret_tensor(buf99, (1, 128, 4096), (524288, 4096, 1), 0); del buf99  # reuse
    cpp_fused_gelu_27(c_void_p(buf100.data_ptr()))
    buf101 = reinterpret_tensor(buf98, (128, 1024), (1024, 1), 0); del buf98  # reuse
    # Source Nodes: [hidden_states_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg65_1, reinterpret_tensor(buf100, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg64_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf101)
    del arg64_1
    del arg65_1
    buf102 = buf96; del buf96  # reuse
    buf103 = buf95; del buf95  # reuse
    buf105 = reinterpret_tensor(buf81, (1, 128, 1024), (131072, 1024, 1), 0); del buf81  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf76.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()))
    del arg66_1
    del arg67_1
    buf106 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg69_1, reinterpret_tensor(buf105, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg68_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf106)
    del arg68_1
    del arg69_1
    buf107 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg71_1, reinterpret_tensor(buf105, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg70_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf107)
    del arg70_1
    del arg71_1
    buf108 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf109 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_29(c_void_p(buf107.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    buf110 = buf91; del buf91  # reuse
    # Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf109, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf108, (16, 64, 128), (8192, 1, 64), 0), out=buf110)
    buf111 = buf88; del buf88  # reuse
    buf112 = buf110; del buf110  # reuse
    buf113 = buf86; del buf86  # reuse
    cpp_fused__softmax_30(c_void_p(buf112.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf113.data_ptr()))
    buf114 = reinterpret_tensor(buf109, (128, 1024), (1024, 1), 0); del buf109  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg73_1, reinterpret_tensor(buf105, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg72_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf114)
    del arg72_1
    del arg73_1
    buf115 = reinterpret_tensor(buf105, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf105  # reuse
    buf116 = buf112; del buf112  # reuse
    cpp_fused__softmax_clone_31(c_void_p(buf116.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()))
    buf117 = reinterpret_tensor(buf114, (16, 128, 64), (8192, 64, 1), 0); del buf114  # reuse
    # Source Nodes: [attn_output_20, attn_weights_19], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf116, reinterpret_tensor(buf115, (16, 128, 64), (8192, 64, 1), 0), out=buf117)
    buf118 = reinterpret_tensor(buf107, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf107  # reuse
    cpp_fused_clone_32(c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()))
    buf119 = reinterpret_tensor(buf117, (128, 1024), (1024, 1), 0); del buf117  # reuse
    # Source Nodes: [hidden_states_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg75_1, reinterpret_tensor(buf118, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg74_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf119)
    del arg74_1
    del arg75_1
    buf120 = buf103; del buf103  # reuse
    buf121 = buf102; del buf102  # reuse
    buf123 = reinterpret_tensor(buf118, (1, 128, 1024), (131072, 1024, 1), 0); del buf118  # reuse
    cpp_fused_add_native_layer_norm_33(c_void_p(buf76.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf123.data_ptr()))
    del arg76_1
    del arg77_1
    buf124 = reinterpret_tensor(buf100, (128, 4096), (4096, 1), 0); del buf100  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg79_1, reinterpret_tensor(buf123, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg78_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf124)
    del arg78_1
    del arg79_1
    buf125 = reinterpret_tensor(buf124, (1, 128, 4096), (524288, 4096, 1), 0); del buf124  # reuse
    cpp_fused_gelu_34(c_void_p(buf125.data_ptr()))
    buf126 = reinterpret_tensor(buf123, (128, 1024), (1024, 1), 0); del buf123  # reuse
    # Source Nodes: [hidden_states_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg81_1, reinterpret_tensor(buf125, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg80_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf126)
    del arg80_1
    del arg81_1
    buf127 = reinterpret_tensor(buf126, (1, 128, 1024), (131072, 1024, 1), 0); del buf126  # reuse
    buf128 = buf121; del buf121  # reuse
    buf129 = buf120; del buf120  # reuse
    buf131 = reinterpret_tensor(buf106, (1, 128, 1024), (131072, 1024, 1), 0); del buf106  # reuse
    cpp_fused_add_native_layer_norm_35(c_void_p(buf127.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf131.data_ptr()))
    del arg82_1
    del arg83_1
    buf132 = buf94; del buf94  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf131, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg84_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf132)
    del arg84_1
    del arg85_1
    buf133 = reinterpret_tensor(buf76, (128, 1024), (1024, 1), 0); del buf76  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg87_1, reinterpret_tensor(buf131, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg86_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf133)
    del arg86_1
    del arg87_1
    buf134 = reinterpret_tensor(buf119, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf119  # reuse
    buf135 = reinterpret_tensor(buf101, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf101  # reuse
    cpp_fused_clone_36(c_void_p(buf133.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    buf136 = buf116; del buf116  # reuse
    # Source Nodes: [attn_weights_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf135, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf134, (16, 64, 128), (8192, 1, 64), 0), out=buf136)
    buf137 = buf113; del buf113  # reuse
    buf138 = buf136; del buf136  # reuse
    buf139 = buf111; del buf111  # reuse
    cpp_fused__softmax_37(c_void_p(buf138.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()))
    buf140 = reinterpret_tensor(buf135, (128, 1024), (1024, 1), 0); del buf135  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg89_1, reinterpret_tensor(buf131, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg88_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf140)
    del arg88_1
    del arg89_1
    buf141 = reinterpret_tensor(buf131, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf131  # reuse
    buf142 = buf138; del buf138  # reuse
    cpp_fused__softmax_clone_38(c_void_p(buf142.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()))
    buf143 = reinterpret_tensor(buf140, (16, 128, 64), (8192, 64, 1), 0); del buf140  # reuse
    # Source Nodes: [attn_output_25, attn_weights_23], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf142, reinterpret_tensor(buf141, (16, 128, 64), (8192, 64, 1), 0), out=buf143)
    buf144 = reinterpret_tensor(buf133, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf133  # reuse
    cpp_fused_clone_39(c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()))
    buf145 = reinterpret_tensor(buf143, (128, 1024), (1024, 1), 0); del buf143  # reuse
    # Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf144, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg90_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf145)
    del arg90_1
    del arg91_1
    buf146 = buf129; del buf129  # reuse
    buf147 = buf128; del buf128  # reuse
    buf149 = reinterpret_tensor(buf144, (1, 128, 1024), (131072, 1024, 1), 0); del buf144  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf127.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()))
    del arg92_1
    del arg93_1
    buf150 = reinterpret_tensor(buf125, (128, 4096), (4096, 1), 0); del buf125  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg95_1, reinterpret_tensor(buf149, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg94_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf150)
    del arg94_1
    del arg95_1
    buf151 = reinterpret_tensor(buf150, (1, 128, 4096), (524288, 4096, 1), 0); del buf150  # reuse
    cpp_fused_gelu_41(c_void_p(buf151.data_ptr()))
    buf152 = reinterpret_tensor(buf149, (128, 1024), (1024, 1), 0); del buf149  # reuse
    # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg97_1, reinterpret_tensor(buf151, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg96_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf152)
    del arg96_1
    del arg97_1
    buf153 = buf147; del buf147  # reuse
    buf154 = buf146; del buf146  # reuse
    buf156 = reinterpret_tensor(buf132, (1, 128, 1024), (131072, 1024, 1), 0); del buf132  # reuse
    cpp_fused_add_native_layer_norm_42(c_void_p(buf127.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()))
    del arg98_1
    del arg99_1
    buf157 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg101_1, reinterpret_tensor(buf156, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg100_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf157)
    del arg100_1
    del arg101_1
    buf158 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg103_1, reinterpret_tensor(buf156, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg102_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf158)
    del arg102_1
    del arg103_1
    buf159 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf160 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_43(c_void_p(buf158.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    buf161 = buf142; del buf142  # reuse
    # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf160, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf159, (16, 64, 128), (8192, 1, 64), 0), out=buf161)
    buf162 = buf139; del buf139  # reuse
    buf163 = buf161; del buf161  # reuse
    buf164 = buf137; del buf137  # reuse
    cpp_fused__softmax_44(c_void_p(buf163.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()))
    buf165 = reinterpret_tensor(buf160, (128, 1024), (1024, 1), 0); del buf160  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg105_1, reinterpret_tensor(buf156, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg104_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf165)
    del arg104_1
    del arg105_1
    buf166 = reinterpret_tensor(buf156, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf156  # reuse
    buf167 = buf163; del buf163  # reuse
    cpp_fused__softmax_clone_45(c_void_p(buf167.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf166.data_ptr()))
    buf168 = reinterpret_tensor(buf165, (16, 128, 64), (8192, 64, 1), 0); del buf165  # reuse
    # Source Nodes: [attn_output_30, attn_weights_27], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf167, reinterpret_tensor(buf166, (16, 128, 64), (8192, 64, 1), 0), out=buf168)
    buf169 = reinterpret_tensor(buf158, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf158  # reuse
    cpp_fused_clone_46(c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    buf170 = reinterpret_tensor(buf168, (128, 1024), (1024, 1), 0); del buf168  # reuse
    # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg107_1, reinterpret_tensor(buf169, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg106_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf170)
    del arg106_1
    del arg107_1
    buf171 = buf154; del buf154  # reuse
    buf172 = buf153; del buf153  # reuse
    buf174 = reinterpret_tensor(buf169, (1, 128, 1024), (131072, 1024, 1), 0); del buf169  # reuse
    cpp_fused_add_native_layer_norm_47(c_void_p(buf127.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()))
    del arg108_1
    del arg109_1
    buf175 = reinterpret_tensor(buf151, (128, 4096), (4096, 1), 0); del buf151  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg111_1, reinterpret_tensor(buf174, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg110_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf175)
    del arg110_1
    del arg111_1
    buf176 = reinterpret_tensor(buf175, (1, 128, 4096), (524288, 4096, 1), 0); del buf175  # reuse
    cpp_fused_gelu_48(c_void_p(buf176.data_ptr()))
    buf177 = reinterpret_tensor(buf174, (128, 1024), (1024, 1), 0); del buf174  # reuse
    # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg113_1, reinterpret_tensor(buf176, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg112_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf177)
    del arg112_1
    del arg113_1
    buf178 = reinterpret_tensor(buf177, (1, 128, 1024), (131072, 1024, 1), 0); del buf177  # reuse
    buf179 = buf172; del buf172  # reuse
    buf180 = buf171; del buf171  # reuse
    buf182 = reinterpret_tensor(buf157, (1, 128, 1024), (131072, 1024, 1), 0); del buf157  # reuse
    cpp_fused_add_native_layer_norm_49(c_void_p(buf178.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()))
    del arg114_1
    del arg115_1
    buf183 = buf170; del buf170  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg117_1, reinterpret_tensor(buf182, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg116_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf183)
    del arg116_1
    del arg117_1
    buf184 = buf152; del buf152  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg119_1, reinterpret_tensor(buf182, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg118_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf184)
    del arg118_1
    del arg119_1
    buf185 = reinterpret_tensor(buf145, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf145  # reuse
    buf186 = reinterpret_tensor(buf127, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf127  # reuse
    cpp_fused_clone_50(c_void_p(buf184.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    buf187 = buf167; del buf167  # reuse
    # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf186, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf185, (16, 64, 128), (8192, 1, 64), 0), out=buf187)
    buf188 = buf164; del buf164  # reuse
    buf189 = buf187; del buf187  # reuse
    buf190 = buf162; del buf162  # reuse
    cpp_fused__softmax_51(c_void_p(buf189.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf190.data_ptr()))
    buf191 = reinterpret_tensor(buf186, (128, 1024), (1024, 1), 0); del buf186  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg121_1, reinterpret_tensor(buf182, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg120_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf191)
    del arg120_1
    del arg121_1
    buf192 = reinterpret_tensor(buf182, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf182  # reuse
    buf193 = buf189; del buf189  # reuse
    cpp_fused__softmax_clone_52(c_void_p(buf193.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf192.data_ptr()))
    buf194 = reinterpret_tensor(buf191, (16, 128, 64), (8192, 64, 1), 0); del buf191  # reuse
    # Source Nodes: [attn_output_35, attn_weights_31], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf193, reinterpret_tensor(buf192, (16, 128, 64), (8192, 64, 1), 0), out=buf194)
    buf195 = reinterpret_tensor(buf184, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf184  # reuse
    cpp_fused_clone_53(c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    buf196 = reinterpret_tensor(buf194, (128, 1024), (1024, 1), 0); del buf194  # reuse
    # Source Nodes: [hidden_states_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg123_1, reinterpret_tensor(buf195, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg122_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf196)
    del arg122_1
    del arg123_1
    buf197 = buf180; del buf180  # reuse
    buf198 = buf179; del buf179  # reuse
    buf200 = reinterpret_tensor(buf195, (1, 128, 1024), (131072, 1024, 1), 0); del buf195  # reuse
    cpp_fused_add_native_layer_norm_54(c_void_p(buf178.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf200.data_ptr()))
    del arg124_1
    del arg125_1
    buf201 = reinterpret_tensor(buf176, (128, 4096), (4096, 1), 0); del buf176  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg127_1, reinterpret_tensor(buf200, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg126_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf201)
    del arg126_1
    del arg127_1
    buf202 = reinterpret_tensor(buf201, (1, 128, 4096), (524288, 4096, 1), 0); del buf201  # reuse
    cpp_fused_gelu_55(c_void_p(buf202.data_ptr()))
    buf203 = reinterpret_tensor(buf200, (128, 1024), (1024, 1), 0); del buf200  # reuse
    # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg129_1, reinterpret_tensor(buf202, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg128_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf203)
    del arg128_1
    del arg129_1
    buf204 = buf198; del buf198  # reuse
    buf205 = buf197; del buf197  # reuse
    buf207 = reinterpret_tensor(buf183, (1, 128, 1024), (131072, 1024, 1), 0); del buf183  # reuse
    cpp_fused_add_native_layer_norm_56(c_void_p(buf178.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()))
    del arg130_1
    del arg131_1
    buf208 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_8_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg133_1, reinterpret_tensor(buf207, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg132_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf208)
    del arg132_1
    del arg133_1
    buf209 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_8_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg135_1, reinterpret_tensor(buf207, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg134_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf209)
    del arg134_1
    del arg135_1
    buf210 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf211 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_57(c_void_p(buf209.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    buf212 = buf193; del buf193  # reuse
    # Source Nodes: [attn_weights_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf211, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf210, (16, 64, 128), (8192, 1, 64), 0), out=buf212)
    buf213 = buf190; del buf190  # reuse
    buf214 = buf212; del buf212  # reuse
    buf215 = buf188; del buf188  # reuse
    cpp_fused__softmax_58(c_void_p(buf214.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()))
    buf216 = reinterpret_tensor(buf211, (128, 1024), (1024, 1), 0); del buf211  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg137_1, reinterpret_tensor(buf207, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg136_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf216)
    del arg136_1
    del arg137_1
    buf217 = reinterpret_tensor(buf207, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf207  # reuse
    buf218 = buf214; del buf214  # reuse
    cpp_fused__softmax_clone_59(c_void_p(buf218.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf217.data_ptr()))
    buf219 = reinterpret_tensor(buf216, (16, 128, 64), (8192, 64, 1), 0); del buf216  # reuse
    # Source Nodes: [attn_output_40, attn_weights_35], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf218, reinterpret_tensor(buf217, (16, 128, 64), (8192, 64, 1), 0), out=buf219)
    buf220 = reinterpret_tensor(buf209, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf209  # reuse
    cpp_fused_clone_60(c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()))
    buf221 = reinterpret_tensor(buf219, (128, 1024), (1024, 1), 0); del buf219  # reuse
    # Source Nodes: [hidden_states_91], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg139_1, reinterpret_tensor(buf220, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg138_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf221)
    del arg138_1
    del arg139_1
    buf222 = buf205; del buf205  # reuse
    buf223 = buf204; del buf204  # reuse
    buf225 = reinterpret_tensor(buf220, (1, 128, 1024), (131072, 1024, 1), 0); del buf220  # reuse
    cpp_fused_add_native_layer_norm_61(c_void_p(buf178.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf225.data_ptr()))
    del arg140_1
    del arg141_1
    buf226 = reinterpret_tensor(buf202, (128, 4096), (4096, 1), 0); del buf202  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, reinterpret_tensor(buf225, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg142_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf226)
    del arg142_1
    del arg143_1
    buf227 = reinterpret_tensor(buf226, (1, 128, 4096), (524288, 4096, 1), 0); del buf226  # reuse
    cpp_fused_gelu_62(c_void_p(buf227.data_ptr()))
    buf228 = reinterpret_tensor(buf225, (128, 1024), (1024, 1), 0); del buf225  # reuse
    # Source Nodes: [hidden_states_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg145_1, reinterpret_tensor(buf227, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg144_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf228)
    del arg144_1
    del arg145_1
    buf229 = reinterpret_tensor(buf228, (1, 128, 1024), (131072, 1024, 1), 0); del buf228  # reuse
    buf230 = buf223; del buf223  # reuse
    buf231 = buf222; del buf222  # reuse
    buf233 = reinterpret_tensor(buf208, (1, 128, 1024), (131072, 1024, 1), 0); del buf208  # reuse
    cpp_fused_add_native_layer_norm_63(c_void_p(buf229.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf233.data_ptr()))
    del arg146_1
    del arg147_1
    buf234 = buf221; del buf221  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg149_1, reinterpret_tensor(buf233, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg148_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf234)
    del arg148_1
    del arg149_1
    buf235 = buf203; del buf203  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg151_1, reinterpret_tensor(buf233, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg150_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf235)
    del arg150_1
    del arg151_1
    buf236 = reinterpret_tensor(buf196, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf196  # reuse
    buf237 = reinterpret_tensor(buf178, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf178  # reuse
    cpp_fused_clone_64(c_void_p(buf235.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()))
    buf238 = buf218; del buf218  # reuse
    # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf237, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf236, (16, 64, 128), (8192, 1, 64), 0), out=buf238)
    buf239 = buf215; del buf215  # reuse
    buf240 = buf238; del buf238  # reuse
    buf241 = buf213; del buf213  # reuse
    cpp_fused__softmax_65(c_void_p(buf240.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf241.data_ptr()))
    buf242 = reinterpret_tensor(buf237, (128, 1024), (1024, 1), 0); del buf237  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg153_1, reinterpret_tensor(buf233, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg152_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf242)
    del arg152_1
    del arg153_1
    buf243 = reinterpret_tensor(buf233, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf233  # reuse
    buf244 = buf240; del buf240  # reuse
    cpp_fused__softmax_clone_66(c_void_p(buf244.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()))
    buf245 = reinterpret_tensor(buf242, (16, 128, 64), (8192, 64, 1), 0); del buf242  # reuse
    # Source Nodes: [attn_output_45, attn_weights_39], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf244, reinterpret_tensor(buf243, (16, 128, 64), (8192, 64, 1), 0), out=buf245)
    buf246 = reinterpret_tensor(buf235, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf235  # reuse
    cpp_fused_clone_67(c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()))
    buf247 = reinterpret_tensor(buf245, (128, 1024), (1024, 1), 0); del buf245  # reuse
    # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg155_1, reinterpret_tensor(buf246, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg154_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf247)
    del arg154_1
    del arg155_1
    buf248 = buf231; del buf231  # reuse
    buf249 = buf230; del buf230  # reuse
    buf251 = reinterpret_tensor(buf246, (1, 128, 1024), (131072, 1024, 1), 0); del buf246  # reuse
    cpp_fused_add_native_layer_norm_68(c_void_p(buf229.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf251.data_ptr()))
    del arg156_1
    del arg157_1
    buf252 = reinterpret_tensor(buf227, (128, 4096), (4096, 1), 0); del buf227  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg159_1, reinterpret_tensor(buf251, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg158_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf252)
    del arg158_1
    del arg159_1
    buf253 = reinterpret_tensor(buf252, (1, 128, 4096), (524288, 4096, 1), 0); del buf252  # reuse
    cpp_fused_gelu_69(c_void_p(buf253.data_ptr()))
    buf254 = reinterpret_tensor(buf251, (128, 1024), (1024, 1), 0); del buf251  # reuse
    # Source Nodes: [hidden_states_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg161_1, reinterpret_tensor(buf253, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg160_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf254)
    del arg160_1
    del arg161_1
    buf255 = buf249; del buf249  # reuse
    buf256 = buf248; del buf248  # reuse
    buf258 = reinterpret_tensor(buf234, (1, 128, 1024), (131072, 1024, 1), 0); del buf234  # reuse
    cpp_fused_add_native_layer_norm_70(c_void_p(buf229.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()))
    del arg162_1
    del arg163_1
    buf259 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_10_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg165_1, reinterpret_tensor(buf258, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg164_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf259)
    del arg164_1
    del arg165_1
    buf260 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_10_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg167_1, reinterpret_tensor(buf258, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg166_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf260)
    del arg166_1
    del arg167_1
    buf261 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf262 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_71(c_void_p(buf260.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()))
    buf263 = buf244; del buf244  # reuse
    # Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf262, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf261, (16, 64, 128), (8192, 1, 64), 0), out=buf263)
    buf264 = buf241; del buf241  # reuse
    buf265 = buf263; del buf263  # reuse
    buf266 = buf239; del buf239  # reuse
    cpp_fused__softmax_72(c_void_p(buf265.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = reinterpret_tensor(buf262, (128, 1024), (1024, 1), 0); del buf262  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg169_1, reinterpret_tensor(buf258, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg168_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf267)
    del arg168_1
    del arg169_1
    buf268 = reinterpret_tensor(buf258, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf258  # reuse
    buf269 = buf265; del buf265  # reuse
    cpp_fused__softmax_clone_73(c_void_p(buf269.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf268.data_ptr()))
    buf270 = reinterpret_tensor(buf267, (16, 128, 64), (8192, 64, 1), 0); del buf267  # reuse
    # Source Nodes: [attn_output_50, attn_weights_43], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf269, reinterpret_tensor(buf268, (16, 128, 64), (8192, 64, 1), 0), out=buf270)
    buf271 = reinterpret_tensor(buf260, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf260  # reuse
    cpp_fused_clone_74(c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    buf272 = reinterpret_tensor(buf270, (128, 1024), (1024, 1), 0); del buf270  # reuse
    # Source Nodes: [hidden_states_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg171_1, reinterpret_tensor(buf271, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg170_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf272)
    del arg170_1
    del arg171_1
    buf273 = buf256; del buf256  # reuse
    buf274 = buf255; del buf255  # reuse
    buf276 = reinterpret_tensor(buf271, (1, 128, 1024), (131072, 1024, 1), 0); del buf271  # reuse
    cpp_fused_add_native_layer_norm_75(c_void_p(buf229.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf276.data_ptr()))
    del arg172_1
    del arg173_1
    buf277 = reinterpret_tensor(buf253, (128, 4096), (4096, 1), 0); del buf253  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg175_1, reinterpret_tensor(buf276, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg174_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf277)
    del arg174_1
    del arg175_1
    buf278 = reinterpret_tensor(buf277, (1, 128, 4096), (524288, 4096, 1), 0); del buf277  # reuse
    cpp_fused_gelu_76(c_void_p(buf278.data_ptr()))
    buf279 = reinterpret_tensor(buf276, (128, 1024), (1024, 1), 0); del buf276  # reuse
    # Source Nodes: [hidden_states_119], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg177_1, reinterpret_tensor(buf278, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg176_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf279)
    del arg176_1
    del arg177_1
    buf280 = reinterpret_tensor(buf279, (1, 128, 1024), (131072, 1024, 1), 0); del buf279  # reuse
    buf281 = buf274; del buf274  # reuse
    buf282 = buf273; del buf273  # reuse
    buf284 = reinterpret_tensor(buf259, (1, 128, 1024), (131072, 1024, 1), 0); del buf259  # reuse
    cpp_fused_add_native_layer_norm_77(c_void_p(buf280.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()))
    del arg178_1
    del arg179_1
    buf285 = buf272; del buf272  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg181_1, reinterpret_tensor(buf284, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg180_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf285)
    del arg180_1
    del arg181_1
    buf286 = buf254; del buf254  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg183_1, reinterpret_tensor(buf284, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg182_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf286)
    del arg182_1
    del arg183_1
    buf287 = reinterpret_tensor(buf247, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf247  # reuse
    buf288 = reinterpret_tensor(buf229, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf229  # reuse
    cpp_fused_clone_78(c_void_p(buf286.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    buf289 = buf269; del buf269  # reuse
    # Source Nodes: [attn_weights_44], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf288, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf287, (16, 64, 128), (8192, 1, 64), 0), out=buf289)
    buf290 = buf266; del buf266  # reuse
    buf291 = buf289; del buf289  # reuse
    buf292 = buf264; del buf264  # reuse
    cpp_fused__softmax_79(c_void_p(buf291.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf292.data_ptr()))
    del buf290
    buf293 = reinterpret_tensor(buf288, (128, 1024), (1024, 1), 0); del buf288  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg185_1, reinterpret_tensor(buf284, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg184_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf293)
    del arg184_1
    del arg185_1
    buf294 = reinterpret_tensor(buf284, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf284  # reuse
    buf295 = buf291; del buf291  # reuse
    cpp_fused__softmax_clone_80(c_void_p(buf295.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf294.data_ptr()))
    del buf292
    buf296 = reinterpret_tensor(buf293, (16, 128, 64), (8192, 64, 1), 0); del buf293  # reuse
    # Source Nodes: [attn_output_55, attn_weights_47], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf295, reinterpret_tensor(buf294, (16, 128, 64), (8192, 64, 1), 0), out=buf296)
    del buf295
    buf297 = reinterpret_tensor(buf286, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf286  # reuse
    cpp_fused_clone_81(c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()))
    buf298 = reinterpret_tensor(buf296, (128, 1024), (1024, 1), 0); del buf296  # reuse
    # Source Nodes: [hidden_states_124], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg187_1, reinterpret_tensor(buf297, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg186_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf298)
    del arg186_1
    del arg187_1
    buf299 = buf282; del buf282  # reuse
    buf300 = buf281; del buf281  # reuse
    buf302 = reinterpret_tensor(buf297, (1, 128, 1024), (131072, 1024, 1), 0); del buf297  # reuse
    cpp_fused_add_native_layer_norm_82(c_void_p(buf280.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf302.data_ptr()))
    del arg188_1
    del arg189_1
    buf303 = reinterpret_tensor(buf278, (128, 4096), (4096, 1), 0); del buf278  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg191_1, reinterpret_tensor(buf302, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg190_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf303)
    del arg190_1
    del arg191_1
    buf304 = reinterpret_tensor(buf303, (1, 128, 4096), (524288, 4096, 1), 0); del buf303  # reuse
    cpp_fused_gelu_83(c_void_p(buf304.data_ptr()))
    buf305 = reinterpret_tensor(buf302, (128, 1024), (1024, 1), 0); del buf302  # reuse
    # Source Nodes: [hidden_states_130], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg193_1, reinterpret_tensor(buf304, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg192_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf305)
    del arg192_1
    del arg193_1
    del buf304
    buf306 = buf300; del buf300  # reuse
    buf307 = buf299; del buf299  # reuse
    buf309 = reinterpret_tensor(buf285, (1, 128, 1024), (131072, 1024, 1), 0); del buf285  # reuse
    cpp_fused_add_native_layer_norm_84(c_void_p(buf280.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf309.data_ptr()))
    del arg194_1
    del arg195_1
    del buf280
    del buf298
    del buf305
    buf310 = empty((128, 50265), device='cpu', dtype=torch.float32)
    # Source Nodes: [logits], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf309, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg196_1, (1024, 50265), (1, 1024), 0), out=buf310)
    del arg196_1
    del buf309
    buf311 = reinterpret_tensor(buf307, (128, 1), (1, 128), 0); del buf307  # reuse
    buf312 = reinterpret_tensor(buf306, (128, 1), (1, 128), 0); del buf306  # reuse
    buf313 = empty((), device='cpu', dtype=torch.float32)
    buf314 = empty((), device='cpu', dtype=torch.int64)
    buf315 = buf313; del buf313  # reuse
    cpp_fused__log_softmax_nll_loss_forward_85(c_void_p(buf315.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf314.data_ptr()))
    del arg198_1
    return (buf315, reinterpret_tensor(buf310, (1, 128, 50265), (6433920, 50265, 1), 0), buf6, buf13, buf32, buf39, buf57, buf64, buf83, buf90, buf108, buf115, buf134, buf141, buf159, buf166, buf185, buf192, buf210, buf217, buf236, buf243, buf261, buf268, buf287, buf294, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((50265, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((50265, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    arg198_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('PegasusForCausalLM', benchmark_compiled_module)
