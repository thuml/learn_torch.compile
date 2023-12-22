
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
                       const long* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 128100);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 128100L), "index out of bounds: 0 <= tmp3 < 128100L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*tmp3)));
                        auto tmp6 = decltype(tmp5)(tmp5 + 512);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 512L), "index out of bounds: 0 <= tmp8 < 512L")
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*tmp8)));
                        auto tmp10 = tmp4 + tmp9;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp10);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = in_ptr2[static_cast<long>(x0)];
                    auto tmp11 = out_ptr0[static_cast<long>(x0)];
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 128100);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 128100L), "index out of bounds: 0 <= tmp3 < 128100L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*tmp3)));
                    auto tmp6 = decltype(tmp5)(tmp5 + 512);
                    auto tmp7 = tmp5 < 0;
                    auto tmp8 = tmp7 ? tmp6 : tmp5;
                    TORCH_CHECK((0 <= tmp8) & (tmp8 < 512L), "index out of bounds: 0 <= tmp8 < 512L")
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*tmp8)));
                    auto tmp10 = tmp4 + tmp9;
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 - tmp12;
                    auto tmp15 = static_cast<float>(1536.0);
                    auto tmp16 = tmp14 / tmp15;
                    auto tmp17 = static_cast<float>(1e-07);
                    auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                    auto tmp19 = 1 / std::sqrt(tmp18);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp13 * tmp20;
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp25 = tmp23 + tmp24;
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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


cpp_fused_add_native_layer_norm_49 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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


cpp_fused_add_native_layer_norm_56 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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


cpp_fused_add_native_layer_norm_63 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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


cpp_fused_add_native_layer_norm_70 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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


cpp_fused_add_native_layer_norm_77 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_79 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_86 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_89 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_94 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_97 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_98 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_100 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_101 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_103 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_104 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_105 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_107 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_108 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_110 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_111 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_112 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_114 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_115 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_117 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_118 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_121 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_122 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_124 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_125 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_126 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_128 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_129 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_132 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_133 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_135 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_136 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_138 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_139 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_140 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_142 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_143 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_145 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_146 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_147 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_148 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_149 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_150 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_151 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_152 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_153 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_154 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_155 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_156 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_157 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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


cpp_fused_add_native_layer_norm_161 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_162 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_163 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_clone_masked_fill_164 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (1536L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_167 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_168 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1536.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-07);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_add_clamp_clone_div_nll_loss_forward_169 = async_compile.cpp('''
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128100, 1536), (1536, 1))
    assert_size_stride(arg1_1, (512, 1536), (1536, 1))
    assert_size_stride(arg2_1, (1536, ), (1, ))
    assert_size_stride(arg3_1, (1536, ), (1, ))
    assert_size_stride(arg4_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg5_1, (1536, ), (1, ))
    assert_size_stride(arg6_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg7_1, (1536, ), (1, ))
    assert_size_stride(arg8_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg9_1, (1536, ), (1, ))
    assert_size_stride(arg10_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg11_1, (1536, ), (1, ))
    assert_size_stride(arg12_1, (1536, ), (1, ))
    assert_size_stride(arg13_1, (1536, ), (1, ))
    assert_size_stride(arg14_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg15_1, (6144, ), (1, ))
    assert_size_stride(arg16_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg17_1, (1536, ), (1, ))
    assert_size_stride(arg18_1, (1536, ), (1, ))
    assert_size_stride(arg19_1, (1536, ), (1, ))
    assert_size_stride(arg20_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg21_1, (1536, ), (1, ))
    assert_size_stride(arg22_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg23_1, (1536, ), (1, ))
    assert_size_stride(arg24_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg25_1, (1536, ), (1, ))
    assert_size_stride(arg26_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg27_1, (1536, ), (1, ))
    assert_size_stride(arg28_1, (1536, ), (1, ))
    assert_size_stride(arg29_1, (1536, ), (1, ))
    assert_size_stride(arg30_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg31_1, (6144, ), (1, ))
    assert_size_stride(arg32_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg33_1, (1536, ), (1, ))
    assert_size_stride(arg34_1, (1536, ), (1, ))
    assert_size_stride(arg35_1, (1536, ), (1, ))
    assert_size_stride(arg36_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg37_1, (1536, ), (1, ))
    assert_size_stride(arg38_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg39_1, (1536, ), (1, ))
    assert_size_stride(arg40_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg41_1, (1536, ), (1, ))
    assert_size_stride(arg42_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg43_1, (1536, ), (1, ))
    assert_size_stride(arg44_1, (1536, ), (1, ))
    assert_size_stride(arg45_1, (1536, ), (1, ))
    assert_size_stride(arg46_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg47_1, (6144, ), (1, ))
    assert_size_stride(arg48_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg49_1, (1536, ), (1, ))
    assert_size_stride(arg50_1, (1536, ), (1, ))
    assert_size_stride(arg51_1, (1536, ), (1, ))
    assert_size_stride(arg52_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg53_1, (1536, ), (1, ))
    assert_size_stride(arg54_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg55_1, (1536, ), (1, ))
    assert_size_stride(arg56_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg57_1, (1536, ), (1, ))
    assert_size_stride(arg58_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg59_1, (1536, ), (1, ))
    assert_size_stride(arg60_1, (1536, ), (1, ))
    assert_size_stride(arg61_1, (1536, ), (1, ))
    assert_size_stride(arg62_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg63_1, (6144, ), (1, ))
    assert_size_stride(arg64_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg65_1, (1536, ), (1, ))
    assert_size_stride(arg66_1, (1536, ), (1, ))
    assert_size_stride(arg67_1, (1536, ), (1, ))
    assert_size_stride(arg68_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg69_1, (1536, ), (1, ))
    assert_size_stride(arg70_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg71_1, (1536, ), (1, ))
    assert_size_stride(arg72_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg73_1, (1536, ), (1, ))
    assert_size_stride(arg74_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg75_1, (1536, ), (1, ))
    assert_size_stride(arg76_1, (1536, ), (1, ))
    assert_size_stride(arg77_1, (1536, ), (1, ))
    assert_size_stride(arg78_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg79_1, (6144, ), (1, ))
    assert_size_stride(arg80_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg81_1, (1536, ), (1, ))
    assert_size_stride(arg82_1, (1536, ), (1, ))
    assert_size_stride(arg83_1, (1536, ), (1, ))
    assert_size_stride(arg84_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg85_1, (1536, ), (1, ))
    assert_size_stride(arg86_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg87_1, (1536, ), (1, ))
    assert_size_stride(arg88_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg89_1, (1536, ), (1, ))
    assert_size_stride(arg90_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg91_1, (1536, ), (1, ))
    assert_size_stride(arg92_1, (1536, ), (1, ))
    assert_size_stride(arg93_1, (1536, ), (1, ))
    assert_size_stride(arg94_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg95_1, (6144, ), (1, ))
    assert_size_stride(arg96_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg97_1, (1536, ), (1, ))
    assert_size_stride(arg98_1, (1536, ), (1, ))
    assert_size_stride(arg99_1, (1536, ), (1, ))
    assert_size_stride(arg100_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg101_1, (1536, ), (1, ))
    assert_size_stride(arg102_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg103_1, (1536, ), (1, ))
    assert_size_stride(arg104_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg105_1, (1536, ), (1, ))
    assert_size_stride(arg106_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg107_1, (1536, ), (1, ))
    assert_size_stride(arg108_1, (1536, ), (1, ))
    assert_size_stride(arg109_1, (1536, ), (1, ))
    assert_size_stride(arg110_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg111_1, (6144, ), (1, ))
    assert_size_stride(arg112_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg113_1, (1536, ), (1, ))
    assert_size_stride(arg114_1, (1536, ), (1, ))
    assert_size_stride(arg115_1, (1536, ), (1, ))
    assert_size_stride(arg116_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg117_1, (1536, ), (1, ))
    assert_size_stride(arg118_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg119_1, (1536, ), (1, ))
    assert_size_stride(arg120_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg121_1, (1536, ), (1, ))
    assert_size_stride(arg122_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg123_1, (1536, ), (1, ))
    assert_size_stride(arg124_1, (1536, ), (1, ))
    assert_size_stride(arg125_1, (1536, ), (1, ))
    assert_size_stride(arg126_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg127_1, (6144, ), (1, ))
    assert_size_stride(arg128_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg129_1, (1536, ), (1, ))
    assert_size_stride(arg130_1, (1536, ), (1, ))
    assert_size_stride(arg131_1, (1536, ), (1, ))
    assert_size_stride(arg132_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg133_1, (1536, ), (1, ))
    assert_size_stride(arg134_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg135_1, (1536, ), (1, ))
    assert_size_stride(arg136_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg137_1, (1536, ), (1, ))
    assert_size_stride(arg138_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg139_1, (1536, ), (1, ))
    assert_size_stride(arg140_1, (1536, ), (1, ))
    assert_size_stride(arg141_1, (1536, ), (1, ))
    assert_size_stride(arg142_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg143_1, (6144, ), (1, ))
    assert_size_stride(arg144_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg145_1, (1536, ), (1, ))
    assert_size_stride(arg146_1, (1536, ), (1, ))
    assert_size_stride(arg147_1, (1536, ), (1, ))
    assert_size_stride(arg148_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg149_1, (1536, ), (1, ))
    assert_size_stride(arg150_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg151_1, (1536, ), (1, ))
    assert_size_stride(arg152_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg153_1, (1536, ), (1, ))
    assert_size_stride(arg154_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg155_1, (1536, ), (1, ))
    assert_size_stride(arg156_1, (1536, ), (1, ))
    assert_size_stride(arg157_1, (1536, ), (1, ))
    assert_size_stride(arg158_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg159_1, (6144, ), (1, ))
    assert_size_stride(arg160_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg161_1, (1536, ), (1, ))
    assert_size_stride(arg162_1, (1536, ), (1, ))
    assert_size_stride(arg163_1, (1536, ), (1, ))
    assert_size_stride(arg164_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg165_1, (1536, ), (1, ))
    assert_size_stride(arg166_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg167_1, (1536, ), (1, ))
    assert_size_stride(arg168_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg169_1, (1536, ), (1, ))
    assert_size_stride(arg170_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg171_1, (1536, ), (1, ))
    assert_size_stride(arg172_1, (1536, ), (1, ))
    assert_size_stride(arg173_1, (1536, ), (1, ))
    assert_size_stride(arg174_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg175_1, (6144, ), (1, ))
    assert_size_stride(arg176_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg177_1, (1536, ), (1, ))
    assert_size_stride(arg178_1, (1536, ), (1, ))
    assert_size_stride(arg179_1, (1536, ), (1, ))
    assert_size_stride(arg180_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg181_1, (1536, ), (1, ))
    assert_size_stride(arg182_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg183_1, (1536, ), (1, ))
    assert_size_stride(arg184_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg185_1, (1536, ), (1, ))
    assert_size_stride(arg186_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg187_1, (1536, ), (1, ))
    assert_size_stride(arg188_1, (1536, ), (1, ))
    assert_size_stride(arg189_1, (1536, ), (1, ))
    assert_size_stride(arg190_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg191_1, (6144, ), (1, ))
    assert_size_stride(arg192_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg193_1, (1536, ), (1, ))
    assert_size_stride(arg194_1, (1536, ), (1, ))
    assert_size_stride(arg195_1, (1536, ), (1, ))
    assert_size_stride(arg196_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg197_1, (1536, ), (1, ))
    assert_size_stride(arg198_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg199_1, (1536, ), (1, ))
    assert_size_stride(arg200_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg201_1, (1536, ), (1, ))
    assert_size_stride(arg202_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg203_1, (1536, ), (1, ))
    assert_size_stride(arg204_1, (1536, ), (1, ))
    assert_size_stride(arg205_1, (1536, ), (1, ))
    assert_size_stride(arg206_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg207_1, (6144, ), (1, ))
    assert_size_stride(arg208_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg209_1, (1536, ), (1, ))
    assert_size_stride(arg210_1, (1536, ), (1, ))
    assert_size_stride(arg211_1, (1536, ), (1, ))
    assert_size_stride(arg212_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg213_1, (1536, ), (1, ))
    assert_size_stride(arg214_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg215_1, (1536, ), (1, ))
    assert_size_stride(arg216_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg217_1, (1536, ), (1, ))
    assert_size_stride(arg218_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg219_1, (1536, ), (1, ))
    assert_size_stride(arg220_1, (1536, ), (1, ))
    assert_size_stride(arg221_1, (1536, ), (1, ))
    assert_size_stride(arg222_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg223_1, (6144, ), (1, ))
    assert_size_stride(arg224_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg225_1, (1536, ), (1, ))
    assert_size_stride(arg226_1, (1536, ), (1, ))
    assert_size_stride(arg227_1, (1536, ), (1, ))
    assert_size_stride(arg228_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg229_1, (1536, ), (1, ))
    assert_size_stride(arg230_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg231_1, (1536, ), (1, ))
    assert_size_stride(arg232_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg233_1, (1536, ), (1, ))
    assert_size_stride(arg234_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg235_1, (1536, ), (1, ))
    assert_size_stride(arg236_1, (1536, ), (1, ))
    assert_size_stride(arg237_1, (1536, ), (1, ))
    assert_size_stride(arg238_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg239_1, (6144, ), (1, ))
    assert_size_stride(arg240_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg241_1, (1536, ), (1, ))
    assert_size_stride(arg242_1, (1536, ), (1, ))
    assert_size_stride(arg243_1, (1536, ), (1, ))
    assert_size_stride(arg244_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg245_1, (1536, ), (1, ))
    assert_size_stride(arg246_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg247_1, (1536, ), (1, ))
    assert_size_stride(arg248_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg249_1, (1536, ), (1, ))
    assert_size_stride(arg250_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg251_1, (1536, ), (1, ))
    assert_size_stride(arg252_1, (1536, ), (1, ))
    assert_size_stride(arg253_1, (1536, ), (1, ))
    assert_size_stride(arg254_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg255_1, (6144, ), (1, ))
    assert_size_stride(arg256_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg257_1, (1536, ), (1, ))
    assert_size_stride(arg258_1, (1536, ), (1, ))
    assert_size_stride(arg259_1, (1536, ), (1, ))
    assert_size_stride(arg260_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg261_1, (1536, ), (1, ))
    assert_size_stride(arg262_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg263_1, (1536, ), (1, ))
    assert_size_stride(arg264_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg265_1, (1536, ), (1, ))
    assert_size_stride(arg266_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg267_1, (1536, ), (1, ))
    assert_size_stride(arg268_1, (1536, ), (1, ))
    assert_size_stride(arg269_1, (1536, ), (1, ))
    assert_size_stride(arg270_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg271_1, (6144, ), (1, ))
    assert_size_stride(arg272_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg273_1, (1536, ), (1, ))
    assert_size_stride(arg274_1, (1536, ), (1, ))
    assert_size_stride(arg275_1, (1536, ), (1, ))
    assert_size_stride(arg276_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg277_1, (1536, ), (1, ))
    assert_size_stride(arg278_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg279_1, (1536, ), (1, ))
    assert_size_stride(arg280_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg281_1, (1536, ), (1, ))
    assert_size_stride(arg282_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg283_1, (1536, ), (1, ))
    assert_size_stride(arg284_1, (1536, ), (1, ))
    assert_size_stride(arg285_1, (1536, ), (1, ))
    assert_size_stride(arg286_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg287_1, (6144, ), (1, ))
    assert_size_stride(arg288_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg289_1, (1536, ), (1, ))
    assert_size_stride(arg290_1, (1536, ), (1, ))
    assert_size_stride(arg291_1, (1536, ), (1, ))
    assert_size_stride(arg292_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg293_1, (1536, ), (1, ))
    assert_size_stride(arg294_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg295_1, (1536, ), (1, ))
    assert_size_stride(arg296_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg297_1, (1536, ), (1, ))
    assert_size_stride(arg298_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg299_1, (1536, ), (1, ))
    assert_size_stride(arg300_1, (1536, ), (1, ))
    assert_size_stride(arg301_1, (1536, ), (1, ))
    assert_size_stride(arg302_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg303_1, (6144, ), (1, ))
    assert_size_stride(arg304_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg305_1, (1536, ), (1, ))
    assert_size_stride(arg306_1, (1536, ), (1, ))
    assert_size_stride(arg307_1, (1536, ), (1, ))
    assert_size_stride(arg308_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg309_1, (1536, ), (1, ))
    assert_size_stride(arg310_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg311_1, (1536, ), (1, ))
    assert_size_stride(arg312_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg313_1, (1536, ), (1, ))
    assert_size_stride(arg314_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg315_1, (1536, ), (1, ))
    assert_size_stride(arg316_1, (1536, ), (1, ))
    assert_size_stride(arg317_1, (1536, ), (1, ))
    assert_size_stride(arg318_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg319_1, (6144, ), (1, ))
    assert_size_stride(arg320_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg321_1, (1536, ), (1, ))
    assert_size_stride(arg322_1, (1536, ), (1, ))
    assert_size_stride(arg323_1, (1536, ), (1, ))
    assert_size_stride(arg324_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg325_1, (1536, ), (1, ))
    assert_size_stride(arg326_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg327_1, (1536, ), (1, ))
    assert_size_stride(arg328_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg329_1, (1536, ), (1, ))
    assert_size_stride(arg330_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg331_1, (1536, ), (1, ))
    assert_size_stride(arg332_1, (1536, ), (1, ))
    assert_size_stride(arg333_1, (1536, ), (1, ))
    assert_size_stride(arg334_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg335_1, (6144, ), (1, ))
    assert_size_stride(arg336_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg337_1, (1536, ), (1, ))
    assert_size_stride(arg338_1, (1536, ), (1, ))
    assert_size_stride(arg339_1, (1536, ), (1, ))
    assert_size_stride(arg340_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg341_1, (1536, ), (1, ))
    assert_size_stride(arg342_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg343_1, (1536, ), (1, ))
    assert_size_stride(arg344_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg345_1, (1536, ), (1, ))
    assert_size_stride(arg346_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg347_1, (1536, ), (1, ))
    assert_size_stride(arg348_1, (1536, ), (1, ))
    assert_size_stride(arg349_1, (1536, ), (1, ))
    assert_size_stride(arg350_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg351_1, (6144, ), (1, ))
    assert_size_stride(arg352_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg353_1, (1536, ), (1, ))
    assert_size_stride(arg354_1, (1536, ), (1, ))
    assert_size_stride(arg355_1, (1536, ), (1, ))
    assert_size_stride(arg356_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg357_1, (1536, ), (1, ))
    assert_size_stride(arg358_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg359_1, (1536, ), (1, ))
    assert_size_stride(arg360_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg361_1, (1536, ), (1, ))
    assert_size_stride(arg362_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg363_1, (1536, ), (1, ))
    assert_size_stride(arg364_1, (1536, ), (1, ))
    assert_size_stride(arg365_1, (1536, ), (1, ))
    assert_size_stride(arg366_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg367_1, (6144, ), (1, ))
    assert_size_stride(arg368_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg369_1, (1536, ), (1, ))
    assert_size_stride(arg370_1, (1536, ), (1, ))
    assert_size_stride(arg371_1, (1536, ), (1, ))
    assert_size_stride(arg372_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg373_1, (1536, ), (1, ))
    assert_size_stride(arg374_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg375_1, (1536, ), (1, ))
    assert_size_stride(arg376_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg377_1, (1536, ), (1, ))
    assert_size_stride(arg378_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg379_1, (1536, ), (1, ))
    assert_size_stride(arg380_1, (1536, ), (1, ))
    assert_size_stride(arg381_1, (1536, ), (1, ))
    assert_size_stride(arg382_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg383_1, (6144, ), (1, ))
    assert_size_stride(arg384_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg385_1, (1536, ), (1, ))
    assert_size_stride(arg386_1, (1536, ), (1, ))
    assert_size_stride(arg387_1, (1536, ), (1, ))
    assert_size_stride(arg388_1, (2, 1536), (1536, 1))
    assert_size_stride(arg389_1, (2, ), (1, ))
    assert_size_stride(arg390_1, (1, 512), (512, 1))
    assert_size_stride(arg391_1, (1, 512), (512, 1))
    assert_size_stride(arg392_1, (1, ), (1, ))
    assert_size_stride(arg393_1, (1, ), (1, ))
    buf0 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mul_native_layer_norm_0(c_void_p(arg391_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg390_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg0_1
    del arg1_1
    del arg2_1
    del arg390_1
    del arg391_1
    del arg3_1
    buf4 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_0_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg5_1, reinterpret_tensor(buf3, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg4_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf4)
    del arg4_1
    del arg5_1
    buf5 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_0_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg7_1, reinterpret_tensor(buf3, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg6_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf5)
    del arg6_1
    del arg7_1
    buf6 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf7 = reinterpret_tensor(buf5, (24, 64, 512), (64, 1, 1536), 0); del buf5  # reuse
    cpp_fused_clone_div_sqrt_1(c_void_p(buf7.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()))
    buf8 = empty((24, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attention_scores, scale, truediv], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf6, (24, 512, 64), (32768, 64, 1), 0), buf7, out=buf8)
    buf9 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf10 = reinterpret_tensor(buf8, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf8  # reuse
    buf11 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_2(c_void_p(buf10.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf11.data_ptr()))
    buf12 = reinterpret_tensor(buf7, (512, 1536), (1536, 1), 0); del buf7  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_0_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg9_1, reinterpret_tensor(buf3, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg8_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf12)
    del arg8_1
    del arg9_1
    buf13 = buf10; del buf10  # reuse
    buf14 = buf6; del buf6  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_3(c_void_p(buf13.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()))
    buf15 = reinterpret_tensor(buf12, (24, 512, 64), (32768, 64, 1), 0); del buf12  # reuse
    # Source Nodes: [context_layer], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf13, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf14, (24, 512, 64), (32768, 64, 1), 0), out=buf15)
    buf16 = reinterpret_tensor(buf14, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf14  # reuse
    cpp_fused_clone_4(c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()))
    buf17 = reinterpret_tensor(buf15, (512, 1536), (1536, 1), 0); del buf15  # reuse
    # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg11_1, reinterpret_tensor(buf16, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg10_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf17)
    del arg10_1
    del arg11_1
    buf18 = buf1; del buf1  # reuse
    buf19 = buf0; del buf0  # reuse
    buf21 = reinterpret_tensor(buf16, (1, 512, 1536), (786432, 1536, 1), 0); del buf16  # reuse
    cpp_fused_add_native_layer_norm_5(c_void_p(buf17.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()))
    del arg12_1
    del arg13_1
    buf22 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg15_1, reinterpret_tensor(buf21, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg14_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf22)
    del arg14_1
    del arg15_1
    buf23 = reinterpret_tensor(buf22, (1, 512, 6144), (3145728, 6144, 1), 0); del buf22  # reuse
    cpp_fused_gelu_6(c_void_p(buf23.data_ptr()))
    buf24 = reinterpret_tensor(buf3, (512, 1536), (1536, 1), 0); del buf3  # reuse
    # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg17_1, reinterpret_tensor(buf23, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg16_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf24)
    del arg16_1
    del arg17_1
    buf25 = buf19; del buf19  # reuse
    buf26 = buf18; del buf18  # reuse
    buf28 = reinterpret_tensor(buf17, (1, 512, 1536), (786432, 1536, 1), 0); del buf17  # reuse
    cpp_fused_add_native_layer_norm_7(c_void_p(buf24.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf28.data_ptr()))
    del arg18_1
    del arg19_1
    buf29 = buf24; del buf24  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_1_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg21_1, reinterpret_tensor(buf28, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg20_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf29)
    del arg20_1
    del arg21_1
    buf30 = reinterpret_tensor(buf21, (512, 1536), (1536, 1), 0); del buf21  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_1_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg23_1, reinterpret_tensor(buf28, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg22_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf30)
    del arg22_1
    del arg23_1
    buf31 = reinterpret_tensor(buf4, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf4  # reuse
    buf32 = reinterpret_tensor(buf30, (24, 64, 512), (64, 1, 1536), 0); del buf30  # reuse
    cpp_fused_clone_div_sqrt_8(c_void_p(buf32.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()))
    buf33 = reinterpret_tensor(buf13, (24, 512, 512), (262144, 512, 1), 0); del buf13  # reuse
    # Source Nodes: [attention_scores_3, scale_1, truediv_1], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf31, (24, 512, 64), (32768, 64, 1), 0), buf32, out=buf33)
    buf34 = buf11; del buf11  # reuse
    buf35 = reinterpret_tensor(buf33, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf33  # reuse
    buf36 = buf9; del buf9  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_9(c_void_p(buf35.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()))
    buf37 = reinterpret_tensor(buf32, (512, 1536), (1536, 1), 0); del buf32  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_1_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg25_1, reinterpret_tensor(buf28, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg24_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf37)
    del arg24_1
    del arg25_1
    buf38 = buf35; del buf35  # reuse
    buf39 = buf31; del buf31  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_10(c_void_p(buf38.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()))
    buf40 = reinterpret_tensor(buf37, (24, 512, 64), (32768, 64, 1), 0); del buf37  # reuse
    # Source Nodes: [context_layer_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf38, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf39, (24, 512, 64), (32768, 64, 1), 0), out=buf40)
    buf41 = reinterpret_tensor(buf39, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf39  # reuse
    cpp_fused_clone_11(c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    buf42 = reinterpret_tensor(buf40, (512, 1536), (1536, 1), 0); del buf40  # reuse
    # Source Nodes: [hidden_states_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg27_1, reinterpret_tensor(buf41, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg26_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf42)
    del arg26_1
    del arg27_1
    buf43 = buf26; del buf26  # reuse
    buf44 = buf25; del buf25  # reuse
    buf46 = reinterpret_tensor(buf41, (1, 512, 1536), (786432, 1536, 1), 0); del buf41  # reuse
    cpp_fused_add_native_layer_norm_12(c_void_p(buf42.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()))
    del arg28_1
    del arg29_1
    buf47 = reinterpret_tensor(buf23, (512, 6144), (6144, 1), 0); del buf23  # reuse
    # Source Nodes: [hidden_states_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg31_1, reinterpret_tensor(buf46, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg30_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf47)
    del arg30_1
    del arg31_1
    buf48 = reinterpret_tensor(buf47, (1, 512, 6144), (3145728, 6144, 1), 0); del buf47  # reuse
    cpp_fused_gelu_13(c_void_p(buf48.data_ptr()))
    buf49 = buf42; del buf42  # reuse
    # Source Nodes: [hidden_states_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg33_1, reinterpret_tensor(buf48, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg32_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf49)
    del arg32_1
    del arg33_1
    buf50 = buf44; del buf44  # reuse
    buf51 = buf43; del buf43  # reuse
    buf53 = buf28; del buf28  # reuse
    cpp_fused_add_native_layer_norm_14(c_void_p(buf49.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()))
    del arg34_1
    del arg35_1
    buf54 = buf49; del buf49  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_2_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg37_1, reinterpret_tensor(buf53, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg36_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf54)
    del arg36_1
    del arg37_1
    buf55 = reinterpret_tensor(buf46, (512, 1536), (1536, 1), 0); del buf46  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_2_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg39_1, reinterpret_tensor(buf53, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg38_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf55)
    del arg38_1
    del arg39_1
    buf56 = reinterpret_tensor(buf29, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf29  # reuse
    buf57 = reinterpret_tensor(buf55, (24, 64, 512), (64, 1, 1536), 0); del buf55  # reuse
    cpp_fused_clone_div_sqrt_15(c_void_p(buf57.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()))
    buf58 = reinterpret_tensor(buf38, (24, 512, 512), (262144, 512, 1), 0); del buf38  # reuse
    # Source Nodes: [attention_scores_6, scale_2, truediv_2], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf56, (24, 512, 64), (32768, 64, 1), 0), buf57, out=buf58)
    buf59 = buf36; del buf36  # reuse
    buf60 = reinterpret_tensor(buf58, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf58  # reuse
    buf61 = buf34; del buf34  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_16(c_void_p(buf60.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()))
    buf62 = reinterpret_tensor(buf57, (512, 1536), (1536, 1), 0); del buf57  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_2_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg41_1, reinterpret_tensor(buf53, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg40_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf62)
    del arg40_1
    del arg41_1
    buf63 = buf60; del buf60  # reuse
    buf64 = buf56; del buf56  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_17(c_void_p(buf63.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()))
    buf65 = reinterpret_tensor(buf62, (24, 512, 64), (32768, 64, 1), 0); del buf62  # reuse
    # Source Nodes: [context_layer_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf63, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf64, (24, 512, 64), (32768, 64, 1), 0), out=buf65)
    buf66 = reinterpret_tensor(buf64, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf64  # reuse
    cpp_fused_clone_18(c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    buf67 = reinterpret_tensor(buf65, (512, 1536), (1536, 1), 0); del buf65  # reuse
    # Source Nodes: [hidden_states_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg43_1, reinterpret_tensor(buf66, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg42_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf67)
    del arg42_1
    del arg43_1
    buf68 = buf51; del buf51  # reuse
    buf69 = buf50; del buf50  # reuse
    buf71 = reinterpret_tensor(buf66, (1, 512, 1536), (786432, 1536, 1), 0); del buf66  # reuse
    cpp_fused_add_native_layer_norm_19(c_void_p(buf67.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()))
    del arg44_1
    del arg45_1
    buf72 = reinterpret_tensor(buf48, (512, 6144), (6144, 1), 0); del buf48  # reuse
    # Source Nodes: [hidden_states_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg47_1, reinterpret_tensor(buf71, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg46_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf72)
    del arg46_1
    del arg47_1
    buf73 = reinterpret_tensor(buf72, (1, 512, 6144), (3145728, 6144, 1), 0); del buf72  # reuse
    cpp_fused_gelu_20(c_void_p(buf73.data_ptr()))
    buf74 = buf67; del buf67  # reuse
    # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg49_1, reinterpret_tensor(buf73, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg48_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf74)
    del arg48_1
    del arg49_1
    buf75 = buf69; del buf69  # reuse
    buf76 = buf68; del buf68  # reuse
    buf78 = buf53; del buf53  # reuse
    cpp_fused_add_native_layer_norm_21(c_void_p(buf74.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()))
    del arg50_1
    del arg51_1
    buf79 = buf74; del buf74  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_3_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg53_1, reinterpret_tensor(buf78, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg52_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf79)
    del arg52_1
    del arg53_1
    buf80 = reinterpret_tensor(buf71, (512, 1536), (1536, 1), 0); del buf71  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_3_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg55_1, reinterpret_tensor(buf78, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg54_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf80)
    del arg54_1
    del arg55_1
    buf81 = reinterpret_tensor(buf54, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf54  # reuse
    buf82 = reinterpret_tensor(buf80, (24, 64, 512), (64, 1, 1536), 0); del buf80  # reuse
    cpp_fused_clone_div_sqrt_22(c_void_p(buf82.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()))
    buf83 = reinterpret_tensor(buf63, (24, 512, 512), (262144, 512, 1), 0); del buf63  # reuse
    # Source Nodes: [attention_scores_9, scale_3, truediv_3], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf81, (24, 512, 64), (32768, 64, 1), 0), buf82, out=buf83)
    buf84 = buf61; del buf61  # reuse
    buf85 = reinterpret_tensor(buf83, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf83  # reuse
    buf86 = buf59; del buf59  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_23(c_void_p(buf85.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()))
    buf87 = reinterpret_tensor(buf82, (512, 1536), (1536, 1), 0); del buf82  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_3_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg57_1, reinterpret_tensor(buf78, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg56_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf87)
    del arg56_1
    del arg57_1
    buf88 = buf85; del buf85  # reuse
    buf89 = buf81; del buf81  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_24(c_void_p(buf88.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()))
    buf90 = reinterpret_tensor(buf87, (24, 512, 64), (32768, 64, 1), 0); del buf87  # reuse
    # Source Nodes: [context_layer_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf88, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf89, (24, 512, 64), (32768, 64, 1), 0), out=buf90)
    buf91 = reinterpret_tensor(buf89, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf89  # reuse
    cpp_fused_clone_25(c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()))
    buf92 = reinterpret_tensor(buf90, (512, 1536), (1536, 1), 0); del buf90  # reuse
    # Source Nodes: [hidden_states_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg59_1, reinterpret_tensor(buf91, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg58_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf92)
    del arg58_1
    del arg59_1
    buf93 = buf76; del buf76  # reuse
    buf94 = buf75; del buf75  # reuse
    buf96 = reinterpret_tensor(buf91, (1, 512, 1536), (786432, 1536, 1), 0); del buf91  # reuse
    cpp_fused_add_native_layer_norm_26(c_void_p(buf92.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()))
    del arg60_1
    del arg61_1
    buf97 = reinterpret_tensor(buf73, (512, 6144), (6144, 1), 0); del buf73  # reuse
    # Source Nodes: [hidden_states_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg63_1, reinterpret_tensor(buf96, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg62_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf97)
    del arg62_1
    del arg63_1
    buf98 = reinterpret_tensor(buf97, (1, 512, 6144), (3145728, 6144, 1), 0); del buf97  # reuse
    cpp_fused_gelu_27(c_void_p(buf98.data_ptr()))
    buf99 = buf92; del buf92  # reuse
    # Source Nodes: [hidden_states_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg65_1, reinterpret_tensor(buf98, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg64_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf99)
    del arg64_1
    del arg65_1
    buf100 = buf94; del buf94  # reuse
    buf101 = buf93; del buf93  # reuse
    buf103 = buf78; del buf78  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf99.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf103.data_ptr()))
    del arg66_1
    del arg67_1
    buf104 = buf99; del buf99  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_4_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg69_1, reinterpret_tensor(buf103, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg68_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf104)
    del arg68_1
    del arg69_1
    buf105 = reinterpret_tensor(buf96, (512, 1536), (1536, 1), 0); del buf96  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_4_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg71_1, reinterpret_tensor(buf103, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg70_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf105)
    del arg70_1
    del arg71_1
    buf106 = reinterpret_tensor(buf79, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf79  # reuse
    buf107 = reinterpret_tensor(buf105, (24, 64, 512), (64, 1, 1536), 0); del buf105  # reuse
    cpp_fused_clone_div_sqrt_29(c_void_p(buf107.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()))
    buf108 = reinterpret_tensor(buf88, (24, 512, 512), (262144, 512, 1), 0); del buf88  # reuse
    # Source Nodes: [attention_scores_12, scale_4, truediv_4], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf106, (24, 512, 64), (32768, 64, 1), 0), buf107, out=buf108)
    buf109 = buf86; del buf86  # reuse
    buf110 = reinterpret_tensor(buf108, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf108  # reuse
    buf111 = buf84; del buf84  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_30(c_void_p(buf110.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()))
    buf112 = reinterpret_tensor(buf107, (512, 1536), (1536, 1), 0); del buf107  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_4_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg73_1, reinterpret_tensor(buf103, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg72_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf112)
    del arg72_1
    del arg73_1
    buf113 = buf110; del buf110  # reuse
    buf114 = buf106; del buf106  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_31(c_void_p(buf113.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()))
    buf115 = reinterpret_tensor(buf112, (24, 512, 64), (32768, 64, 1), 0); del buf112  # reuse
    # Source Nodes: [context_layer_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf113, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf114, (24, 512, 64), (32768, 64, 1), 0), out=buf115)
    buf116 = reinterpret_tensor(buf114, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf114  # reuse
    cpp_fused_clone_32(c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    buf117 = reinterpret_tensor(buf115, (512, 1536), (1536, 1), 0); del buf115  # reuse
    # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg75_1, reinterpret_tensor(buf116, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg74_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf117)
    del arg74_1
    del arg75_1
    buf118 = buf101; del buf101  # reuse
    buf119 = buf100; del buf100  # reuse
    buf121 = reinterpret_tensor(buf116, (1, 512, 1536), (786432, 1536, 1), 0); del buf116  # reuse
    cpp_fused_add_native_layer_norm_33(c_void_p(buf117.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()))
    del arg76_1
    del arg77_1
    buf122 = reinterpret_tensor(buf98, (512, 6144), (6144, 1), 0); del buf98  # reuse
    # Source Nodes: [hidden_states_34], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg79_1, reinterpret_tensor(buf121, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg78_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf122)
    del arg78_1
    del arg79_1
    buf123 = reinterpret_tensor(buf122, (1, 512, 6144), (3145728, 6144, 1), 0); del buf122  # reuse
    cpp_fused_gelu_34(c_void_p(buf123.data_ptr()))
    buf124 = buf117; del buf117  # reuse
    # Source Nodes: [hidden_states_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg81_1, reinterpret_tensor(buf123, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg80_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf124)
    del arg80_1
    del arg81_1
    buf125 = buf119; del buf119  # reuse
    buf126 = buf118; del buf118  # reuse
    buf128 = buf103; del buf103  # reuse
    cpp_fused_add_native_layer_norm_35(c_void_p(buf124.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()))
    del arg82_1
    del arg83_1
    buf129 = buf124; del buf124  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_5_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf128, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg84_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf129)
    del arg84_1
    del arg85_1
    buf130 = reinterpret_tensor(buf121, (512, 1536), (1536, 1), 0); del buf121  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_5_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg87_1, reinterpret_tensor(buf128, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg86_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf130)
    del arg86_1
    del arg87_1
    buf131 = reinterpret_tensor(buf104, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf104  # reuse
    buf132 = reinterpret_tensor(buf130, (24, 64, 512), (64, 1, 1536), 0); del buf130  # reuse
    cpp_fused_clone_div_sqrt_36(c_void_p(buf132.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf131.data_ptr()))
    buf133 = reinterpret_tensor(buf113, (24, 512, 512), (262144, 512, 1), 0); del buf113  # reuse
    # Source Nodes: [attention_scores_15, scale_5, truediv_5], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf131, (24, 512, 64), (32768, 64, 1), 0), buf132, out=buf133)
    buf134 = buf111; del buf111  # reuse
    buf135 = reinterpret_tensor(buf133, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf133  # reuse
    buf136 = buf109; del buf109  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_37(c_void_p(buf135.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()))
    buf137 = reinterpret_tensor(buf132, (512, 1536), (1536, 1), 0); del buf132  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_5_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg89_1, reinterpret_tensor(buf128, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg88_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf137)
    del arg88_1
    del arg89_1
    buf138 = buf135; del buf135  # reuse
    buf139 = buf131; del buf131  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_38(c_void_p(buf138.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()))
    buf140 = reinterpret_tensor(buf137, (24, 512, 64), (32768, 64, 1), 0); del buf137  # reuse
    # Source Nodes: [context_layer_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf138, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf139, (24, 512, 64), (32768, 64, 1), 0), out=buf140)
    buf141 = reinterpret_tensor(buf139, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf139  # reuse
    cpp_fused_clone_39(c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    buf142 = reinterpret_tensor(buf140, (512, 1536), (1536, 1), 0); del buf140  # reuse
    # Source Nodes: [hidden_states_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf141, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg90_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf142)
    del arg90_1
    del arg91_1
    buf143 = buf126; del buf126  # reuse
    buf144 = buf125; del buf125  # reuse
    buf146 = reinterpret_tensor(buf141, (1, 512, 1536), (786432, 1536, 1), 0); del buf141  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf142.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()))
    del arg92_1
    del arg93_1
    buf147 = reinterpret_tensor(buf123, (512, 6144), (6144, 1), 0); del buf123  # reuse
    # Source Nodes: [hidden_states_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg95_1, reinterpret_tensor(buf146, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg94_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf147)
    del arg94_1
    del arg95_1
    buf148 = reinterpret_tensor(buf147, (1, 512, 6144), (3145728, 6144, 1), 0); del buf147  # reuse
    cpp_fused_gelu_41(c_void_p(buf148.data_ptr()))
    buf149 = buf142; del buf142  # reuse
    # Source Nodes: [hidden_states_45], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg97_1, reinterpret_tensor(buf148, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg96_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf149)
    del arg96_1
    del arg97_1
    buf150 = buf144; del buf144  # reuse
    buf151 = buf143; del buf143  # reuse
    buf153 = buf128; del buf128  # reuse
    cpp_fused_add_native_layer_norm_42(c_void_p(buf149.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()))
    del arg98_1
    del arg99_1
    buf154 = buf149; del buf149  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_6_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg101_1, reinterpret_tensor(buf153, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg100_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf154)
    del arg100_1
    del arg101_1
    buf155 = reinterpret_tensor(buf146, (512, 1536), (1536, 1), 0); del buf146  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_6_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg103_1, reinterpret_tensor(buf153, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg102_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf155)
    del arg102_1
    del arg103_1
    buf156 = reinterpret_tensor(buf129, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf129  # reuse
    buf157 = reinterpret_tensor(buf155, (24, 64, 512), (64, 1, 1536), 0); del buf155  # reuse
    cpp_fused_clone_div_sqrt_43(c_void_p(buf157.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()))
    buf158 = reinterpret_tensor(buf138, (24, 512, 512), (262144, 512, 1), 0); del buf138  # reuse
    # Source Nodes: [attention_scores_18, scale_6, truediv_6], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf156, (24, 512, 64), (32768, 64, 1), 0), buf157, out=buf158)
    buf159 = buf136; del buf136  # reuse
    buf160 = reinterpret_tensor(buf158, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf158  # reuse
    buf161 = buf134; del buf134  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_44(c_void_p(buf160.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()))
    buf162 = reinterpret_tensor(buf157, (512, 1536), (1536, 1), 0); del buf157  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_6_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg105_1, reinterpret_tensor(buf153, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg104_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf162)
    del arg104_1
    del arg105_1
    buf163 = buf160; del buf160  # reuse
    buf164 = buf156; del buf156  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_45(c_void_p(buf163.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()))
    buf165 = reinterpret_tensor(buf162, (24, 512, 64), (32768, 64, 1), 0); del buf162  # reuse
    # Source Nodes: [context_layer_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf163, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf164, (24, 512, 64), (32768, 64, 1), 0), out=buf165)
    buf166 = reinterpret_tensor(buf164, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf164  # reuse
    cpp_fused_clone_46(c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    buf167 = reinterpret_tensor(buf165, (512, 1536), (1536, 1), 0); del buf165  # reuse
    # Source Nodes: [hidden_states_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg107_1, reinterpret_tensor(buf166, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg106_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf167)
    del arg106_1
    del arg107_1
    buf168 = buf151; del buf151  # reuse
    buf169 = buf150; del buf150  # reuse
    buf171 = reinterpret_tensor(buf166, (1, 512, 1536), (786432, 1536, 1), 0); del buf166  # reuse
    cpp_fused_add_native_layer_norm_47(c_void_p(buf167.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()))
    del arg108_1
    del arg109_1
    buf172 = reinterpret_tensor(buf148, (512, 6144), (6144, 1), 0); del buf148  # reuse
    # Source Nodes: [hidden_states_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg111_1, reinterpret_tensor(buf171, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg110_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf172)
    del arg110_1
    del arg111_1
    buf173 = reinterpret_tensor(buf172, (1, 512, 6144), (3145728, 6144, 1), 0); del buf172  # reuse
    cpp_fused_gelu_48(c_void_p(buf173.data_ptr()))
    buf174 = buf167; del buf167  # reuse
    # Source Nodes: [hidden_states_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg113_1, reinterpret_tensor(buf173, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg112_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf174)
    del arg112_1
    del arg113_1
    buf175 = buf169; del buf169  # reuse
    buf176 = buf168; del buf168  # reuse
    buf178 = buf153; del buf153  # reuse
    cpp_fused_add_native_layer_norm_49(c_void_p(buf174.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf178.data_ptr()))
    del arg114_1
    del arg115_1
    buf179 = buf174; del buf174  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_7_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg117_1, reinterpret_tensor(buf178, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg116_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf179)
    del arg116_1
    del arg117_1
    buf180 = reinterpret_tensor(buf171, (512, 1536), (1536, 1), 0); del buf171  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_7_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg119_1, reinterpret_tensor(buf178, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg118_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf180)
    del arg118_1
    del arg119_1
    buf181 = reinterpret_tensor(buf154, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf154  # reuse
    buf182 = reinterpret_tensor(buf180, (24, 64, 512), (64, 1, 1536), 0); del buf180  # reuse
    cpp_fused_clone_div_sqrt_50(c_void_p(buf182.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()))
    buf183 = reinterpret_tensor(buf163, (24, 512, 512), (262144, 512, 1), 0); del buf163  # reuse
    # Source Nodes: [attention_scores_21, scale_7, truediv_7], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf181, (24, 512, 64), (32768, 64, 1), 0), buf182, out=buf183)
    buf184 = buf161; del buf161  # reuse
    buf185 = reinterpret_tensor(buf183, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf183  # reuse
    buf186 = buf159; del buf159  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_51(c_void_p(buf185.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()))
    buf187 = reinterpret_tensor(buf182, (512, 1536), (1536, 1), 0); del buf182  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_7_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg121_1, reinterpret_tensor(buf178, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg120_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf187)
    del arg120_1
    del arg121_1
    buf188 = buf185; del buf185  # reuse
    buf189 = buf181; del buf181  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_52(c_void_p(buf188.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf189.data_ptr()))
    buf190 = reinterpret_tensor(buf187, (24, 512, 64), (32768, 64, 1), 0); del buf187  # reuse
    # Source Nodes: [context_layer_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf188, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf189, (24, 512, 64), (32768, 64, 1), 0), out=buf190)
    buf191 = reinterpret_tensor(buf189, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf189  # reuse
    cpp_fused_clone_53(c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    buf192 = reinterpret_tensor(buf190, (512, 1536), (1536, 1), 0); del buf190  # reuse
    # Source Nodes: [hidden_states_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg123_1, reinterpret_tensor(buf191, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg122_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf192)
    del arg122_1
    del arg123_1
    buf193 = buf176; del buf176  # reuse
    buf194 = buf175; del buf175  # reuse
    buf196 = reinterpret_tensor(buf191, (1, 512, 1536), (786432, 1536, 1), 0); del buf191  # reuse
    cpp_fused_add_native_layer_norm_54(c_void_p(buf192.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf196.data_ptr()))
    del arg124_1
    del arg125_1
    buf197 = reinterpret_tensor(buf173, (512, 6144), (6144, 1), 0); del buf173  # reuse
    # Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg127_1, reinterpret_tensor(buf196, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg126_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf197)
    del arg126_1
    del arg127_1
    buf198 = reinterpret_tensor(buf197, (1, 512, 6144), (3145728, 6144, 1), 0); del buf197  # reuse
    cpp_fused_gelu_55(c_void_p(buf198.data_ptr()))
    buf199 = buf192; del buf192  # reuse
    # Source Nodes: [hidden_states_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg129_1, reinterpret_tensor(buf198, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg128_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf199)
    del arg128_1
    del arg129_1
    buf200 = buf194; del buf194  # reuse
    buf201 = buf193; del buf193  # reuse
    buf203 = buf178; del buf178  # reuse
    cpp_fused_add_native_layer_norm_56(c_void_p(buf199.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf203.data_ptr()))
    del arg130_1
    del arg131_1
    buf204 = buf199; del buf199  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_8_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg133_1, reinterpret_tensor(buf203, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg132_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf204)
    del arg132_1
    del arg133_1
    buf205 = reinterpret_tensor(buf196, (512, 1536), (1536, 1), 0); del buf196  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_8_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg135_1, reinterpret_tensor(buf203, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg134_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf205)
    del arg134_1
    del arg135_1
    buf206 = reinterpret_tensor(buf179, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf179  # reuse
    buf207 = reinterpret_tensor(buf205, (24, 64, 512), (64, 1, 1536), 0); del buf205  # reuse
    cpp_fused_clone_div_sqrt_57(c_void_p(buf207.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf206.data_ptr()))
    buf208 = reinterpret_tensor(buf188, (24, 512, 512), (262144, 512, 1), 0); del buf188  # reuse
    # Source Nodes: [attention_scores_24, scale_8, truediv_8], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf206, (24, 512, 64), (32768, 64, 1), 0), buf207, out=buf208)
    buf209 = buf186; del buf186  # reuse
    buf210 = reinterpret_tensor(buf208, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf208  # reuse
    buf211 = buf184; del buf184  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_58(c_void_p(buf210.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()))
    buf212 = reinterpret_tensor(buf207, (512, 1536), (1536, 1), 0); del buf207  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_8_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg137_1, reinterpret_tensor(buf203, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg136_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf212)
    del arg136_1
    del arg137_1
    buf213 = buf210; del buf210  # reuse
    buf214 = buf206; del buf206  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_59(c_void_p(buf213.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf214.data_ptr()))
    buf215 = reinterpret_tensor(buf212, (24, 512, 64), (32768, 64, 1), 0); del buf212  # reuse
    # Source Nodes: [context_layer_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf213, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf214, (24, 512, 64), (32768, 64, 1), 0), out=buf215)
    buf216 = reinterpret_tensor(buf214, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf214  # reuse
    cpp_fused_clone_60(c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    buf217 = reinterpret_tensor(buf215, (512, 1536), (1536, 1), 0); del buf215  # reuse
    # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg139_1, reinterpret_tensor(buf216, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg138_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf217)
    del arg138_1
    del arg139_1
    buf218 = buf201; del buf201  # reuse
    buf219 = buf200; del buf200  # reuse
    buf221 = reinterpret_tensor(buf216, (1, 512, 1536), (786432, 1536, 1), 0); del buf216  # reuse
    cpp_fused_add_native_layer_norm_61(c_void_p(buf217.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf221.data_ptr()))
    del arg140_1
    del arg141_1
    buf222 = reinterpret_tensor(buf198, (512, 6144), (6144, 1), 0); del buf198  # reuse
    # Source Nodes: [hidden_states_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, reinterpret_tensor(buf221, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg142_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf222)
    del arg142_1
    del arg143_1
    buf223 = reinterpret_tensor(buf222, (1, 512, 6144), (3145728, 6144, 1), 0); del buf222  # reuse
    cpp_fused_gelu_62(c_void_p(buf223.data_ptr()))
    buf224 = buf217; del buf217  # reuse
    # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg145_1, reinterpret_tensor(buf223, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg144_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf224)
    del arg144_1
    del arg145_1
    buf225 = buf219; del buf219  # reuse
    buf226 = buf218; del buf218  # reuse
    buf228 = buf203; del buf203  # reuse
    cpp_fused_add_native_layer_norm_63(c_void_p(buf224.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()))
    del arg146_1
    del arg147_1
    buf229 = buf224; del buf224  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_9_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg149_1, reinterpret_tensor(buf228, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg148_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf229)
    del arg148_1
    del arg149_1
    buf230 = reinterpret_tensor(buf221, (512, 1536), (1536, 1), 0); del buf221  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_9_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg151_1, reinterpret_tensor(buf228, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg150_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf230)
    del arg150_1
    del arg151_1
    buf231 = reinterpret_tensor(buf204, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf204  # reuse
    buf232 = reinterpret_tensor(buf230, (24, 64, 512), (64, 1, 1536), 0); del buf230  # reuse
    cpp_fused_clone_div_sqrt_64(c_void_p(buf232.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()))
    buf233 = reinterpret_tensor(buf213, (24, 512, 512), (262144, 512, 1), 0); del buf213  # reuse
    # Source Nodes: [attention_scores_27, scale_9, truediv_9], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf231, (24, 512, 64), (32768, 64, 1), 0), buf232, out=buf233)
    buf234 = buf211; del buf211  # reuse
    buf235 = reinterpret_tensor(buf233, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf233  # reuse
    buf236 = buf209; del buf209  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_65(c_void_p(buf235.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf236.data_ptr()))
    buf237 = reinterpret_tensor(buf232, (512, 1536), (1536, 1), 0); del buf232  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_9_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg153_1, reinterpret_tensor(buf228, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg152_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf237)
    del arg152_1
    del arg153_1
    buf238 = buf235; del buf235  # reuse
    buf239 = buf231; del buf231  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_66(c_void_p(buf238.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()))
    buf240 = reinterpret_tensor(buf237, (24, 512, 64), (32768, 64, 1), 0); del buf237  # reuse
    # Source Nodes: [context_layer_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf238, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf239, (24, 512, 64), (32768, 64, 1), 0), out=buf240)
    buf241 = reinterpret_tensor(buf239, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf239  # reuse
    cpp_fused_clone_67(c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    buf242 = reinterpret_tensor(buf240, (512, 1536), (1536, 1), 0); del buf240  # reuse
    # Source Nodes: [hidden_states_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg155_1, reinterpret_tensor(buf241, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg154_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf242)
    del arg154_1
    del arg155_1
    buf243 = buf226; del buf226  # reuse
    buf244 = buf225; del buf225  # reuse
    buf246 = reinterpret_tensor(buf241, (1, 512, 1536), (786432, 1536, 1), 0); del buf241  # reuse
    cpp_fused_add_native_layer_norm_68(c_void_p(buf242.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()))
    del arg156_1
    del arg157_1
    buf247 = reinterpret_tensor(buf223, (512, 6144), (6144, 1), 0); del buf223  # reuse
    # Source Nodes: [hidden_states_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg159_1, reinterpret_tensor(buf246, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg158_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf247)
    del arg158_1
    del arg159_1
    buf248 = reinterpret_tensor(buf247, (1, 512, 6144), (3145728, 6144, 1), 0); del buf247  # reuse
    cpp_fused_gelu_69(c_void_p(buf248.data_ptr()))
    buf249 = buf242; del buf242  # reuse
    # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg161_1, reinterpret_tensor(buf248, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg160_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf249)
    del arg160_1
    del arg161_1
    buf250 = buf244; del buf244  # reuse
    buf251 = buf243; del buf243  # reuse
    buf253 = buf228; del buf228  # reuse
    cpp_fused_add_native_layer_norm_70(c_void_p(buf249.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()))
    del arg162_1
    del arg163_1
    buf254 = buf249; del buf249  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_10_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg165_1, reinterpret_tensor(buf253, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg164_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf254)
    del arg164_1
    del arg165_1
    buf255 = reinterpret_tensor(buf246, (512, 1536), (1536, 1), 0); del buf246  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_10_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg167_1, reinterpret_tensor(buf253, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg166_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf255)
    del arg166_1
    del arg167_1
    buf256 = reinterpret_tensor(buf229, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf229  # reuse
    buf257 = reinterpret_tensor(buf255, (24, 64, 512), (64, 1, 1536), 0); del buf255  # reuse
    cpp_fused_clone_div_sqrt_71(c_void_p(buf257.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf256.data_ptr()))
    buf258 = reinterpret_tensor(buf238, (24, 512, 512), (262144, 512, 1), 0); del buf238  # reuse
    # Source Nodes: [attention_scores_30, scale_10, truediv_10], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf256, (24, 512, 64), (32768, 64, 1), 0), buf257, out=buf258)
    buf259 = buf236; del buf236  # reuse
    buf260 = reinterpret_tensor(buf258, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf258  # reuse
    buf261 = buf234; del buf234  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_72(c_void_p(buf260.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()))
    buf262 = reinterpret_tensor(buf257, (512, 1536), (1536, 1), 0); del buf257  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_10_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg169_1, reinterpret_tensor(buf253, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg168_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf262)
    del arg168_1
    del arg169_1
    buf263 = buf260; del buf260  # reuse
    buf264 = buf256; del buf256  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_73(c_void_p(buf263.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()))
    buf265 = reinterpret_tensor(buf262, (24, 512, 64), (32768, 64, 1), 0); del buf262  # reuse
    # Source Nodes: [context_layer_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf263, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf264, (24, 512, 64), (32768, 64, 1), 0), out=buf265)
    buf266 = reinterpret_tensor(buf264, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf264  # reuse
    cpp_fused_clone_74(c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = reinterpret_tensor(buf265, (512, 1536), (1536, 1), 0); del buf265  # reuse
    # Source Nodes: [hidden_states_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg171_1, reinterpret_tensor(buf266, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg170_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf267)
    del arg170_1
    del arg171_1
    buf268 = buf251; del buf251  # reuse
    buf269 = buf250; del buf250  # reuse
    buf271 = reinterpret_tensor(buf266, (1, 512, 1536), (786432, 1536, 1), 0); del buf266  # reuse
    cpp_fused_add_native_layer_norm_75(c_void_p(buf267.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf271.data_ptr()))
    del arg172_1
    del arg173_1
    buf272 = reinterpret_tensor(buf248, (512, 6144), (6144, 1), 0); del buf248  # reuse
    # Source Nodes: [hidden_states_82], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg175_1, reinterpret_tensor(buf271, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg174_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf272)
    del arg174_1
    del arg175_1
    buf273 = reinterpret_tensor(buf272, (1, 512, 6144), (3145728, 6144, 1), 0); del buf272  # reuse
    cpp_fused_gelu_76(c_void_p(buf273.data_ptr()))
    buf274 = buf267; del buf267  # reuse
    # Source Nodes: [hidden_states_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg177_1, reinterpret_tensor(buf273, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg176_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf274)
    del arg176_1
    del arg177_1
    buf275 = buf269; del buf269  # reuse
    buf276 = buf268; del buf268  # reuse
    buf278 = buf253; del buf253  # reuse
    cpp_fused_add_native_layer_norm_77(c_void_p(buf274.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf278.data_ptr()))
    del arg178_1
    del arg179_1
    buf279 = buf274; del buf274  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_11_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg181_1, reinterpret_tensor(buf278, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg180_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf279)
    del arg180_1
    del arg181_1
    buf280 = reinterpret_tensor(buf271, (512, 1536), (1536, 1), 0); del buf271  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_11_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg183_1, reinterpret_tensor(buf278, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg182_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf280)
    del arg182_1
    del arg183_1
    buf281 = reinterpret_tensor(buf254, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf254  # reuse
    buf282 = reinterpret_tensor(buf280, (24, 64, 512), (64, 1, 1536), 0); del buf280  # reuse
    cpp_fused_clone_div_sqrt_78(c_void_p(buf282.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()))
    buf283 = reinterpret_tensor(buf263, (24, 512, 512), (262144, 512, 1), 0); del buf263  # reuse
    # Source Nodes: [attention_scores_33, scale_11, truediv_11], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf281, (24, 512, 64), (32768, 64, 1), 0), buf282, out=buf283)
    buf284 = buf261; del buf261  # reuse
    buf285 = reinterpret_tensor(buf283, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf283  # reuse
    buf286 = buf259; del buf259  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_79(c_void_p(buf285.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf286.data_ptr()))
    buf287 = reinterpret_tensor(buf282, (512, 1536), (1536, 1), 0); del buf282  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_11_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg185_1, reinterpret_tensor(buf278, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg184_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf287)
    del arg184_1
    del arg185_1
    buf288 = buf285; del buf285  # reuse
    buf289 = buf281; del buf281  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_80(c_void_p(buf288.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()))
    buf290 = reinterpret_tensor(buf287, (24, 512, 64), (32768, 64, 1), 0); del buf287  # reuse
    # Source Nodes: [context_layer_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf288, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf289, (24, 512, 64), (32768, 64, 1), 0), out=buf290)
    buf291 = reinterpret_tensor(buf289, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf289  # reuse
    cpp_fused_clone_81(c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    buf292 = reinterpret_tensor(buf290, (512, 1536), (1536, 1), 0); del buf290  # reuse
    # Source Nodes: [hidden_states_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg187_1, reinterpret_tensor(buf291, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg186_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf292)
    del arg186_1
    del arg187_1
    buf293 = buf276; del buf276  # reuse
    buf294 = buf275; del buf275  # reuse
    buf296 = reinterpret_tensor(buf291, (1, 512, 1536), (786432, 1536, 1), 0); del buf291  # reuse
    cpp_fused_add_native_layer_norm_82(c_void_p(buf292.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()))
    del arg188_1
    del arg189_1
    buf297 = reinterpret_tensor(buf273, (512, 6144), (6144, 1), 0); del buf273  # reuse
    # Source Nodes: [hidden_states_90], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg191_1, reinterpret_tensor(buf296, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg190_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf297)
    del arg190_1
    del arg191_1
    buf298 = reinterpret_tensor(buf297, (1, 512, 6144), (3145728, 6144, 1), 0); del buf297  # reuse
    cpp_fused_gelu_83(c_void_p(buf298.data_ptr()))
    buf299 = buf292; del buf292  # reuse
    # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg193_1, reinterpret_tensor(buf298, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg192_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf299)
    del arg192_1
    del arg193_1
    buf300 = buf294; del buf294  # reuse
    buf301 = buf293; del buf293  # reuse
    buf303 = buf278; del buf278  # reuse
    cpp_fused_add_native_layer_norm_84(c_void_p(buf299.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf303.data_ptr()))
    del arg194_1
    del arg195_1
    buf304 = buf299; del buf299  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_12_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg197_1, reinterpret_tensor(buf303, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg196_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf304)
    del arg196_1
    del arg197_1
    buf305 = reinterpret_tensor(buf296, (512, 1536), (1536, 1), 0); del buf296  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_12_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg199_1, reinterpret_tensor(buf303, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg198_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf305)
    del arg198_1
    del arg199_1
    buf306 = reinterpret_tensor(buf279, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf279  # reuse
    buf307 = reinterpret_tensor(buf305, (24, 64, 512), (64, 1, 1536), 0); del buf305  # reuse
    cpp_fused_clone_div_sqrt_85(c_void_p(buf307.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()))
    buf308 = reinterpret_tensor(buf288, (24, 512, 512), (262144, 512, 1), 0); del buf288  # reuse
    # Source Nodes: [attention_scores_36, scale_12, truediv_12], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf306, (24, 512, 64), (32768, 64, 1), 0), buf307, out=buf308)
    buf309 = buf286; del buf286  # reuse
    buf310 = reinterpret_tensor(buf308, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf308  # reuse
    buf311 = buf284; del buf284  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_86(c_void_p(buf310.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()))
    buf312 = reinterpret_tensor(buf307, (512, 1536), (1536, 1), 0); del buf307  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_12_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg201_1, reinterpret_tensor(buf303, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg200_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf312)
    del arg200_1
    del arg201_1
    buf313 = buf310; del buf310  # reuse
    buf314 = buf306; del buf306  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_87(c_void_p(buf313.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf314.data_ptr()))
    buf315 = reinterpret_tensor(buf312, (24, 512, 64), (32768, 64, 1), 0); del buf312  # reuse
    # Source Nodes: [context_layer_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf313, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf314, (24, 512, 64), (32768, 64, 1), 0), out=buf315)
    buf316 = reinterpret_tensor(buf314, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf314  # reuse
    cpp_fused_clone_88(c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()))
    buf317 = reinterpret_tensor(buf315, (512, 1536), (1536, 1), 0); del buf315  # reuse
    # Source Nodes: [hidden_states_96], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg203_1, reinterpret_tensor(buf316, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg202_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf317)
    del arg202_1
    del arg203_1
    buf318 = buf301; del buf301  # reuse
    buf319 = buf300; del buf300  # reuse
    buf321 = reinterpret_tensor(buf316, (1, 512, 1536), (786432, 1536, 1), 0); del buf316  # reuse
    cpp_fused_add_native_layer_norm_89(c_void_p(buf317.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf321.data_ptr()))
    del arg204_1
    del arg205_1
    buf322 = reinterpret_tensor(buf298, (512, 6144), (6144, 1), 0); del buf298  # reuse
    # Source Nodes: [hidden_states_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg207_1, reinterpret_tensor(buf321, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg206_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf322)
    del arg206_1
    del arg207_1
    buf323 = reinterpret_tensor(buf322, (1, 512, 6144), (3145728, 6144, 1), 0); del buf322  # reuse
    cpp_fused_gelu_90(c_void_p(buf323.data_ptr()))
    buf324 = buf317; del buf317  # reuse
    # Source Nodes: [hidden_states_101], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg209_1, reinterpret_tensor(buf323, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg208_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf324)
    del arg208_1
    del arg209_1
    buf325 = buf319; del buf319  # reuse
    buf326 = buf318; del buf318  # reuse
    buf328 = buf303; del buf303  # reuse
    cpp_fused_add_native_layer_norm_91(c_void_p(buf324.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf328.data_ptr()))
    del arg210_1
    del arg211_1
    buf329 = buf324; del buf324  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_13_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg213_1, reinterpret_tensor(buf328, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg212_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf329)
    del arg212_1
    del arg213_1
    buf330 = reinterpret_tensor(buf321, (512, 1536), (1536, 1), 0); del buf321  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_13_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg215_1, reinterpret_tensor(buf328, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg214_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf330)
    del arg214_1
    del arg215_1
    buf331 = reinterpret_tensor(buf304, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf304  # reuse
    buf332 = reinterpret_tensor(buf330, (24, 64, 512), (64, 1, 1536), 0); del buf330  # reuse
    cpp_fused_clone_div_sqrt_92(c_void_p(buf332.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf331.data_ptr()))
    buf333 = reinterpret_tensor(buf313, (24, 512, 512), (262144, 512, 1), 0); del buf313  # reuse
    # Source Nodes: [attention_scores_39, scale_13, truediv_13], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf331, (24, 512, 64), (32768, 64, 1), 0), buf332, out=buf333)
    buf334 = buf311; del buf311  # reuse
    buf335 = reinterpret_tensor(buf333, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf333  # reuse
    buf336 = buf309; del buf309  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_93(c_void_p(buf335.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()))
    buf337 = reinterpret_tensor(buf332, (512, 1536), (1536, 1), 0); del buf332  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_13_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg217_1, reinterpret_tensor(buf328, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg216_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf337)
    del arg216_1
    del arg217_1
    buf338 = buf335; del buf335  # reuse
    buf339 = buf331; del buf331  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_94(c_void_p(buf338.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf339.data_ptr()))
    buf340 = reinterpret_tensor(buf337, (24, 512, 64), (32768, 64, 1), 0); del buf337  # reuse
    # Source Nodes: [context_layer_39], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf338, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf339, (24, 512, 64), (32768, 64, 1), 0), out=buf340)
    buf341 = reinterpret_tensor(buf339, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf339  # reuse
    cpp_fused_clone_95(c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()))
    buf342 = reinterpret_tensor(buf340, (512, 1536), (1536, 1), 0); del buf340  # reuse
    # Source Nodes: [hidden_states_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg219_1, reinterpret_tensor(buf341, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg218_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf342)
    del arg218_1
    del arg219_1
    buf343 = buf326; del buf326  # reuse
    buf344 = buf325; del buf325  # reuse
    buf346 = reinterpret_tensor(buf341, (1, 512, 1536), (786432, 1536, 1), 0); del buf341  # reuse
    cpp_fused_add_native_layer_norm_96(c_void_p(buf342.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf346.data_ptr()))
    del arg220_1
    del arg221_1
    buf347 = reinterpret_tensor(buf323, (512, 6144), (6144, 1), 0); del buf323  # reuse
    # Source Nodes: [hidden_states_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg223_1, reinterpret_tensor(buf346, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg222_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf347)
    del arg222_1
    del arg223_1
    buf348 = reinterpret_tensor(buf347, (1, 512, 6144), (3145728, 6144, 1), 0); del buf347  # reuse
    cpp_fused_gelu_97(c_void_p(buf348.data_ptr()))
    buf349 = buf342; del buf342  # reuse
    # Source Nodes: [hidden_states_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg225_1, reinterpret_tensor(buf348, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg224_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf349)
    del arg224_1
    del arg225_1
    buf350 = buf344; del buf344  # reuse
    buf351 = buf343; del buf343  # reuse
    buf353 = buf328; del buf328  # reuse
    cpp_fused_add_native_layer_norm_98(c_void_p(buf349.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf353.data_ptr()))
    del arg226_1
    del arg227_1
    buf354 = buf349; del buf349  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_14_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg229_1, reinterpret_tensor(buf353, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg228_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf354)
    del arg228_1
    del arg229_1
    buf355 = reinterpret_tensor(buf346, (512, 1536), (1536, 1), 0); del buf346  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_14_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg231_1, reinterpret_tensor(buf353, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg230_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf355)
    del arg230_1
    del arg231_1
    buf356 = reinterpret_tensor(buf329, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf329  # reuse
    buf357 = reinterpret_tensor(buf355, (24, 64, 512), (64, 1, 1536), 0); del buf355  # reuse
    cpp_fused_clone_div_sqrt_99(c_void_p(buf357.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf356.data_ptr()))
    buf358 = reinterpret_tensor(buf338, (24, 512, 512), (262144, 512, 1), 0); del buf338  # reuse
    # Source Nodes: [attention_scores_42, scale_14, truediv_14], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf356, (24, 512, 64), (32768, 64, 1), 0), buf357, out=buf358)
    buf359 = buf336; del buf336  # reuse
    buf360 = reinterpret_tensor(buf358, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf358  # reuse
    buf361 = buf334; del buf334  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_100(c_void_p(buf360.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf361.data_ptr()))
    buf362 = reinterpret_tensor(buf357, (512, 1536), (1536, 1), 0); del buf357  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_14_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg233_1, reinterpret_tensor(buf353, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg232_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf362)
    del arg232_1
    del arg233_1
    buf363 = buf360; del buf360  # reuse
    buf364 = buf356; del buf356  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_101(c_void_p(buf363.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf364.data_ptr()))
    buf365 = reinterpret_tensor(buf362, (24, 512, 64), (32768, 64, 1), 0); del buf362  # reuse
    # Source Nodes: [context_layer_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf363, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf364, (24, 512, 64), (32768, 64, 1), 0), out=buf365)
    buf366 = reinterpret_tensor(buf364, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf364  # reuse
    cpp_fused_clone_102(c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()))
    buf367 = reinterpret_tensor(buf365, (512, 1536), (1536, 1), 0); del buf365  # reuse
    # Source Nodes: [hidden_states_112], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg235_1, reinterpret_tensor(buf366, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg234_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf367)
    del arg234_1
    del arg235_1
    buf368 = buf351; del buf351  # reuse
    buf369 = buf350; del buf350  # reuse
    buf371 = reinterpret_tensor(buf366, (1, 512, 1536), (786432, 1536, 1), 0); del buf366  # reuse
    cpp_fused_add_native_layer_norm_103(c_void_p(buf367.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf371.data_ptr()))
    del arg236_1
    del arg237_1
    buf372 = reinterpret_tensor(buf348, (512, 6144), (6144, 1), 0); del buf348  # reuse
    # Source Nodes: [hidden_states_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg239_1, reinterpret_tensor(buf371, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg238_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf372)
    del arg238_1
    del arg239_1
    buf373 = reinterpret_tensor(buf372, (1, 512, 6144), (3145728, 6144, 1), 0); del buf372  # reuse
    cpp_fused_gelu_104(c_void_p(buf373.data_ptr()))
    buf374 = buf367; del buf367  # reuse
    # Source Nodes: [hidden_states_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg241_1, reinterpret_tensor(buf373, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg240_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf374)
    del arg240_1
    del arg241_1
    buf375 = buf369; del buf369  # reuse
    buf376 = buf368; del buf368  # reuse
    buf378 = buf353; del buf353  # reuse
    cpp_fused_add_native_layer_norm_105(c_void_p(buf374.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf378.data_ptr()))
    del arg242_1
    del arg243_1
    buf379 = buf374; del buf374  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_15_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg245_1, reinterpret_tensor(buf378, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg244_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf379)
    del arg244_1
    del arg245_1
    buf380 = reinterpret_tensor(buf371, (512, 1536), (1536, 1), 0); del buf371  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_15_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg247_1, reinterpret_tensor(buf378, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg246_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf380)
    del arg246_1
    del arg247_1
    buf381 = reinterpret_tensor(buf354, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf354  # reuse
    buf382 = reinterpret_tensor(buf380, (24, 64, 512), (64, 1, 1536), 0); del buf380  # reuse
    cpp_fused_clone_div_sqrt_106(c_void_p(buf382.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf381.data_ptr()))
    buf383 = reinterpret_tensor(buf363, (24, 512, 512), (262144, 512, 1), 0); del buf363  # reuse
    # Source Nodes: [attention_scores_45, scale_15, truediv_15], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf381, (24, 512, 64), (32768, 64, 1), 0), buf382, out=buf383)
    buf384 = buf361; del buf361  # reuse
    buf385 = reinterpret_tensor(buf383, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf383  # reuse
    buf386 = buf359; del buf359  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_107(c_void_p(buf385.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf386.data_ptr()))
    buf387 = reinterpret_tensor(buf382, (512, 1536), (1536, 1), 0); del buf382  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_15_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg249_1, reinterpret_tensor(buf378, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg248_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf387)
    del arg248_1
    del arg249_1
    buf388 = buf385; del buf385  # reuse
    buf389 = buf381; del buf381  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_108(c_void_p(buf388.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf389.data_ptr()))
    buf390 = reinterpret_tensor(buf387, (24, 512, 64), (32768, 64, 1), 0); del buf387  # reuse
    # Source Nodes: [context_layer_45], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf388, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf389, (24, 512, 64), (32768, 64, 1), 0), out=buf390)
    buf391 = reinterpret_tensor(buf389, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf389  # reuse
    cpp_fused_clone_109(c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()))
    buf392 = reinterpret_tensor(buf390, (512, 1536), (1536, 1), 0); del buf390  # reuse
    # Source Nodes: [hidden_states_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg251_1, reinterpret_tensor(buf391, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg250_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf392)
    del arg250_1
    del arg251_1
    buf393 = buf376; del buf376  # reuse
    buf394 = buf375; del buf375  # reuse
    buf396 = reinterpret_tensor(buf391, (1, 512, 1536), (786432, 1536, 1), 0); del buf391  # reuse
    cpp_fused_add_native_layer_norm_110(c_void_p(buf392.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf396.data_ptr()))
    del arg252_1
    del arg253_1
    buf397 = reinterpret_tensor(buf373, (512, 6144), (6144, 1), 0); del buf373  # reuse
    # Source Nodes: [hidden_states_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg255_1, reinterpret_tensor(buf396, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg254_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf397)
    del arg254_1
    del arg255_1
    buf398 = reinterpret_tensor(buf397, (1, 512, 6144), (3145728, 6144, 1), 0); del buf397  # reuse
    cpp_fused_gelu_111(c_void_p(buf398.data_ptr()))
    buf399 = buf392; del buf392  # reuse
    # Source Nodes: [hidden_states_125], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg257_1, reinterpret_tensor(buf398, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg256_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf399)
    del arg256_1
    del arg257_1
    buf400 = buf394; del buf394  # reuse
    buf401 = buf393; del buf393  # reuse
    buf403 = buf378; del buf378  # reuse
    cpp_fused_add_native_layer_norm_112(c_void_p(buf399.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf403.data_ptr()))
    del arg258_1
    del arg259_1
    buf404 = buf399; del buf399  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_16_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg261_1, reinterpret_tensor(buf403, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg260_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf404)
    del arg260_1
    del arg261_1
    buf405 = reinterpret_tensor(buf396, (512, 1536), (1536, 1), 0); del buf396  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_16_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg263_1, reinterpret_tensor(buf403, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg262_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf405)
    del arg262_1
    del arg263_1
    buf406 = reinterpret_tensor(buf379, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf379  # reuse
    buf407 = reinterpret_tensor(buf405, (24, 64, 512), (64, 1, 1536), 0); del buf405  # reuse
    cpp_fused_clone_div_sqrt_113(c_void_p(buf407.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf406.data_ptr()))
    buf408 = reinterpret_tensor(buf388, (24, 512, 512), (262144, 512, 1), 0); del buf388  # reuse
    # Source Nodes: [attention_scores_48, scale_16, truediv_16], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf406, (24, 512, 64), (32768, 64, 1), 0), buf407, out=buf408)
    buf409 = buf386; del buf386  # reuse
    buf410 = reinterpret_tensor(buf408, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf408  # reuse
    buf411 = buf384; del buf384  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_114(c_void_p(buf410.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf411.data_ptr()))
    buf412 = reinterpret_tensor(buf407, (512, 1536), (1536, 1), 0); del buf407  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_16_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg265_1, reinterpret_tensor(buf403, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg264_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf412)
    del arg264_1
    del arg265_1
    buf413 = buf410; del buf410  # reuse
    buf414 = buf406; del buf406  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_115(c_void_p(buf413.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf414.data_ptr()))
    buf415 = reinterpret_tensor(buf412, (24, 512, 64), (32768, 64, 1), 0); del buf412  # reuse
    # Source Nodes: [context_layer_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf413, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf414, (24, 512, 64), (32768, 64, 1), 0), out=buf415)
    buf416 = reinterpret_tensor(buf414, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf414  # reuse
    cpp_fused_clone_116(c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()))
    buf417 = reinterpret_tensor(buf415, (512, 1536), (1536, 1), 0); del buf415  # reuse
    # Source Nodes: [hidden_states_128], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg267_1, reinterpret_tensor(buf416, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg266_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf417)
    del arg266_1
    del arg267_1
    buf418 = buf401; del buf401  # reuse
    buf419 = buf400; del buf400  # reuse
    buf421 = reinterpret_tensor(buf416, (1, 512, 1536), (786432, 1536, 1), 0); del buf416  # reuse
    cpp_fused_add_native_layer_norm_117(c_void_p(buf417.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf421.data_ptr()))
    del arg268_1
    del arg269_1
    buf422 = reinterpret_tensor(buf398, (512, 6144), (6144, 1), 0); del buf398  # reuse
    # Source Nodes: [hidden_states_130], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg271_1, reinterpret_tensor(buf421, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg270_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf422)
    del arg270_1
    del arg271_1
    buf423 = reinterpret_tensor(buf422, (1, 512, 6144), (3145728, 6144, 1), 0); del buf422  # reuse
    cpp_fused_gelu_118(c_void_p(buf423.data_ptr()))
    buf424 = buf417; del buf417  # reuse
    # Source Nodes: [hidden_states_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg273_1, reinterpret_tensor(buf423, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg272_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf424)
    del arg272_1
    del arg273_1
    buf425 = buf419; del buf419  # reuse
    buf426 = buf418; del buf418  # reuse
    buf428 = buf403; del buf403  # reuse
    cpp_fused_add_native_layer_norm_119(c_void_p(buf424.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf428.data_ptr()))
    del arg274_1
    del arg275_1
    buf429 = buf424; del buf424  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_17_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg277_1, reinterpret_tensor(buf428, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg276_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf429)
    del arg276_1
    del arg277_1
    buf430 = reinterpret_tensor(buf421, (512, 1536), (1536, 1), 0); del buf421  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_17_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg279_1, reinterpret_tensor(buf428, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg278_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf430)
    del arg278_1
    del arg279_1
    buf431 = reinterpret_tensor(buf404, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf404  # reuse
    buf432 = reinterpret_tensor(buf430, (24, 64, 512), (64, 1, 1536), 0); del buf430  # reuse
    cpp_fused_clone_div_sqrt_120(c_void_p(buf432.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf431.data_ptr()))
    buf433 = reinterpret_tensor(buf413, (24, 512, 512), (262144, 512, 1), 0); del buf413  # reuse
    # Source Nodes: [attention_scores_51, scale_17, truediv_17], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf431, (24, 512, 64), (32768, 64, 1), 0), buf432, out=buf433)
    buf434 = buf411; del buf411  # reuse
    buf435 = reinterpret_tensor(buf433, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf433  # reuse
    buf436 = buf409; del buf409  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_121(c_void_p(buf435.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf436.data_ptr()))
    buf437 = reinterpret_tensor(buf432, (512, 1536), (1536, 1), 0); del buf432  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_17_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg281_1, reinterpret_tensor(buf428, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg280_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf437)
    del arg280_1
    del arg281_1
    buf438 = buf435; del buf435  # reuse
    buf439 = buf431; del buf431  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_122(c_void_p(buf438.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf439.data_ptr()))
    buf440 = reinterpret_tensor(buf437, (24, 512, 64), (32768, 64, 1), 0); del buf437  # reuse
    # Source Nodes: [context_layer_51], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf438, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf439, (24, 512, 64), (32768, 64, 1), 0), out=buf440)
    buf441 = reinterpret_tensor(buf439, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf439  # reuse
    cpp_fused_clone_123(c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()))
    buf442 = reinterpret_tensor(buf440, (512, 1536), (1536, 1), 0); del buf440  # reuse
    # Source Nodes: [hidden_states_136], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg283_1, reinterpret_tensor(buf441, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg282_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf442)
    del arg282_1
    del arg283_1
    buf443 = buf426; del buf426  # reuse
    buf444 = buf425; del buf425  # reuse
    buf446 = reinterpret_tensor(buf441, (1, 512, 1536), (786432, 1536, 1), 0); del buf441  # reuse
    cpp_fused_add_native_layer_norm_124(c_void_p(buf442.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf446.data_ptr()))
    del arg284_1
    del arg285_1
    buf447 = reinterpret_tensor(buf423, (512, 6144), (6144, 1), 0); del buf423  # reuse
    # Source Nodes: [hidden_states_138], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg287_1, reinterpret_tensor(buf446, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg286_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf447)
    del arg286_1
    del arg287_1
    buf448 = reinterpret_tensor(buf447, (1, 512, 6144), (3145728, 6144, 1), 0); del buf447  # reuse
    cpp_fused_gelu_125(c_void_p(buf448.data_ptr()))
    buf449 = buf442; del buf442  # reuse
    # Source Nodes: [hidden_states_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg289_1, reinterpret_tensor(buf448, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg288_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf449)
    del arg288_1
    del arg289_1
    buf450 = buf444; del buf444  # reuse
    buf451 = buf443; del buf443  # reuse
    buf453 = buf428; del buf428  # reuse
    cpp_fused_add_native_layer_norm_126(c_void_p(buf449.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf453.data_ptr()))
    del arg290_1
    del arg291_1
    buf454 = buf449; del buf449  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_18_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg293_1, reinterpret_tensor(buf453, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg292_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf454)
    del arg292_1
    del arg293_1
    buf455 = reinterpret_tensor(buf446, (512, 1536), (1536, 1), 0); del buf446  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_18_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg295_1, reinterpret_tensor(buf453, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg294_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf455)
    del arg294_1
    del arg295_1
    buf456 = reinterpret_tensor(buf429, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf429  # reuse
    buf457 = reinterpret_tensor(buf455, (24, 64, 512), (64, 1, 1536), 0); del buf455  # reuse
    cpp_fused_clone_div_sqrt_127(c_void_p(buf457.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf456.data_ptr()))
    buf458 = reinterpret_tensor(buf438, (24, 512, 512), (262144, 512, 1), 0); del buf438  # reuse
    # Source Nodes: [attention_scores_54, scale_18, truediv_18], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf456, (24, 512, 64), (32768, 64, 1), 0), buf457, out=buf458)
    buf459 = buf436; del buf436  # reuse
    buf460 = reinterpret_tensor(buf458, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf458  # reuse
    buf461 = buf434; del buf434  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_128(c_void_p(buf460.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf461.data_ptr()))
    buf462 = reinterpret_tensor(buf457, (512, 1536), (1536, 1), 0); del buf457  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_18_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg297_1, reinterpret_tensor(buf453, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg296_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf462)
    del arg296_1
    del arg297_1
    buf463 = buf460; del buf460  # reuse
    buf464 = buf456; del buf456  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_129(c_void_p(buf463.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf464.data_ptr()))
    buf465 = reinterpret_tensor(buf462, (24, 512, 64), (32768, 64, 1), 0); del buf462  # reuse
    # Source Nodes: [context_layer_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf463, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf464, (24, 512, 64), (32768, 64, 1), 0), out=buf465)
    buf466 = reinterpret_tensor(buf464, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf464  # reuse
    cpp_fused_clone_130(c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()))
    buf467 = reinterpret_tensor(buf465, (512, 1536), (1536, 1), 0); del buf465  # reuse
    # Source Nodes: [hidden_states_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg299_1, reinterpret_tensor(buf466, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg298_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf467)
    del arg298_1
    del arg299_1
    buf468 = buf451; del buf451  # reuse
    buf469 = buf450; del buf450  # reuse
    buf471 = reinterpret_tensor(buf466, (1, 512, 1536), (786432, 1536, 1), 0); del buf466  # reuse
    cpp_fused_add_native_layer_norm_131(c_void_p(buf467.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf471.data_ptr()))
    del arg300_1
    del arg301_1
    buf472 = reinterpret_tensor(buf448, (512, 6144), (6144, 1), 0); del buf448  # reuse
    # Source Nodes: [hidden_states_146], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg303_1, reinterpret_tensor(buf471, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg302_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf472)
    del arg302_1
    del arg303_1
    buf473 = reinterpret_tensor(buf472, (1, 512, 6144), (3145728, 6144, 1), 0); del buf472  # reuse
    cpp_fused_gelu_132(c_void_p(buf473.data_ptr()))
    buf474 = buf467; del buf467  # reuse
    # Source Nodes: [hidden_states_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg305_1, reinterpret_tensor(buf473, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg304_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf474)
    del arg304_1
    del arg305_1
    buf475 = buf469; del buf469  # reuse
    buf476 = buf468; del buf468  # reuse
    buf478 = buf453; del buf453  # reuse
    cpp_fused_add_native_layer_norm_133(c_void_p(buf474.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf478.data_ptr()))
    del arg306_1
    del arg307_1
    buf479 = buf474; del buf474  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_19_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg309_1, reinterpret_tensor(buf478, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg308_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf479)
    del arg308_1
    del arg309_1
    buf480 = reinterpret_tensor(buf471, (512, 1536), (1536, 1), 0); del buf471  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_19_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg311_1, reinterpret_tensor(buf478, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg310_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf480)
    del arg310_1
    del arg311_1
    buf481 = reinterpret_tensor(buf454, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf454  # reuse
    buf482 = reinterpret_tensor(buf480, (24, 64, 512), (64, 1, 1536), 0); del buf480  # reuse
    cpp_fused_clone_div_sqrt_134(c_void_p(buf482.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf481.data_ptr()))
    buf483 = reinterpret_tensor(buf463, (24, 512, 512), (262144, 512, 1), 0); del buf463  # reuse
    # Source Nodes: [attention_scores_57, scale_19, truediv_19], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf481, (24, 512, 64), (32768, 64, 1), 0), buf482, out=buf483)
    buf484 = buf461; del buf461  # reuse
    buf485 = reinterpret_tensor(buf483, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf483  # reuse
    buf486 = buf459; del buf459  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_135(c_void_p(buf485.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf486.data_ptr()))
    buf487 = reinterpret_tensor(buf482, (512, 1536), (1536, 1), 0); del buf482  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_19_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg313_1, reinterpret_tensor(buf478, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg312_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf487)
    del arg312_1
    del arg313_1
    buf488 = buf485; del buf485  # reuse
    buf489 = buf481; del buf481  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_136(c_void_p(buf488.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf489.data_ptr()))
    buf490 = reinterpret_tensor(buf487, (24, 512, 64), (32768, 64, 1), 0); del buf487  # reuse
    # Source Nodes: [context_layer_57], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf488, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf489, (24, 512, 64), (32768, 64, 1), 0), out=buf490)
    buf491 = reinterpret_tensor(buf489, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf489  # reuse
    cpp_fused_clone_137(c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()))
    buf492 = reinterpret_tensor(buf490, (512, 1536), (1536, 1), 0); del buf490  # reuse
    # Source Nodes: [hidden_states_152], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg315_1, reinterpret_tensor(buf491, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg314_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf492)
    del arg314_1
    del arg315_1
    buf493 = buf476; del buf476  # reuse
    buf494 = buf475; del buf475  # reuse
    buf496 = reinterpret_tensor(buf491, (1, 512, 1536), (786432, 1536, 1), 0); del buf491  # reuse
    cpp_fused_add_native_layer_norm_138(c_void_p(buf492.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf496.data_ptr()))
    del arg316_1
    del arg317_1
    buf497 = reinterpret_tensor(buf473, (512, 6144), (6144, 1), 0); del buf473  # reuse
    # Source Nodes: [hidden_states_154], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg319_1, reinterpret_tensor(buf496, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg318_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf497)
    del arg318_1
    del arg319_1
    buf498 = reinterpret_tensor(buf497, (1, 512, 6144), (3145728, 6144, 1), 0); del buf497  # reuse
    cpp_fused_gelu_139(c_void_p(buf498.data_ptr()))
    buf499 = buf492; del buf492  # reuse
    # Source Nodes: [hidden_states_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg321_1, reinterpret_tensor(buf498, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg320_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf499)
    del arg320_1
    del arg321_1
    buf500 = buf494; del buf494  # reuse
    buf501 = buf493; del buf493  # reuse
    buf503 = buf478; del buf478  # reuse
    cpp_fused_add_native_layer_norm_140(c_void_p(buf499.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf503.data_ptr()))
    del arg322_1
    del arg323_1
    buf504 = buf499; del buf499  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_20_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg325_1, reinterpret_tensor(buf503, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg324_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf504)
    del arg324_1
    del arg325_1
    buf505 = reinterpret_tensor(buf496, (512, 1536), (1536, 1), 0); del buf496  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_20_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg327_1, reinterpret_tensor(buf503, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg326_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf505)
    del arg326_1
    del arg327_1
    buf506 = reinterpret_tensor(buf479, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf479  # reuse
    buf507 = reinterpret_tensor(buf505, (24, 64, 512), (64, 1, 1536), 0); del buf505  # reuse
    cpp_fused_clone_div_sqrt_141(c_void_p(buf507.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf506.data_ptr()))
    buf508 = reinterpret_tensor(buf488, (24, 512, 512), (262144, 512, 1), 0); del buf488  # reuse
    # Source Nodes: [attention_scores_60, scale_20, truediv_20], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf506, (24, 512, 64), (32768, 64, 1), 0), buf507, out=buf508)
    buf509 = buf486; del buf486  # reuse
    buf510 = reinterpret_tensor(buf508, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf508  # reuse
    buf511 = buf484; del buf484  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_142(c_void_p(buf510.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf511.data_ptr()))
    buf512 = reinterpret_tensor(buf507, (512, 1536), (1536, 1), 0); del buf507  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_20_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg329_1, reinterpret_tensor(buf503, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg328_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf512)
    del arg328_1
    del arg329_1
    buf513 = buf510; del buf510  # reuse
    buf514 = buf506; del buf506  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_143(c_void_p(buf513.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf514.data_ptr()))
    buf515 = reinterpret_tensor(buf512, (24, 512, 64), (32768, 64, 1), 0); del buf512  # reuse
    # Source Nodes: [context_layer_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf513, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf514, (24, 512, 64), (32768, 64, 1), 0), out=buf515)
    buf516 = reinterpret_tensor(buf514, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf514  # reuse
    cpp_fused_clone_144(c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()))
    buf517 = reinterpret_tensor(buf515, (512, 1536), (1536, 1), 0); del buf515  # reuse
    # Source Nodes: [hidden_states_160], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg331_1, reinterpret_tensor(buf516, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg330_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf517)
    del arg330_1
    del arg331_1
    buf518 = buf501; del buf501  # reuse
    buf519 = buf500; del buf500  # reuse
    buf521 = reinterpret_tensor(buf516, (1, 512, 1536), (786432, 1536, 1), 0); del buf516  # reuse
    cpp_fused_add_native_layer_norm_145(c_void_p(buf517.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf521.data_ptr()))
    del arg332_1
    del arg333_1
    buf522 = reinterpret_tensor(buf498, (512, 6144), (6144, 1), 0); del buf498  # reuse
    # Source Nodes: [hidden_states_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg335_1, reinterpret_tensor(buf521, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg334_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf522)
    del arg334_1
    del arg335_1
    buf523 = reinterpret_tensor(buf522, (1, 512, 6144), (3145728, 6144, 1), 0); del buf522  # reuse
    cpp_fused_gelu_146(c_void_p(buf523.data_ptr()))
    buf524 = buf517; del buf517  # reuse
    # Source Nodes: [hidden_states_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg337_1, reinterpret_tensor(buf523, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg336_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf524)
    del arg336_1
    del arg337_1
    buf525 = buf519; del buf519  # reuse
    buf526 = buf518; del buf518  # reuse
    buf528 = buf503; del buf503  # reuse
    cpp_fused_add_native_layer_norm_147(c_void_p(buf524.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf528.data_ptr()))
    del arg338_1
    del arg339_1
    buf529 = buf524; del buf524  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_21_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg341_1, reinterpret_tensor(buf528, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg340_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf529)
    del arg340_1
    del arg341_1
    buf530 = reinterpret_tensor(buf521, (512, 1536), (1536, 1), 0); del buf521  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_21_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg343_1, reinterpret_tensor(buf528, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg342_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf530)
    del arg342_1
    del arg343_1
    buf531 = reinterpret_tensor(buf504, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf504  # reuse
    buf532 = reinterpret_tensor(buf530, (24, 64, 512), (64, 1, 1536), 0); del buf530  # reuse
    cpp_fused_clone_div_sqrt_148(c_void_p(buf532.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf531.data_ptr()))
    buf533 = reinterpret_tensor(buf513, (24, 512, 512), (262144, 512, 1), 0); del buf513  # reuse
    # Source Nodes: [attention_scores_63, scale_21, truediv_21], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf531, (24, 512, 64), (32768, 64, 1), 0), buf532, out=buf533)
    buf534 = buf511; del buf511  # reuse
    buf535 = reinterpret_tensor(buf533, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf533  # reuse
    buf536 = buf509; del buf509  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_149(c_void_p(buf535.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf536.data_ptr()))
    buf537 = reinterpret_tensor(buf532, (512, 1536), (1536, 1), 0); del buf532  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_21_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg345_1, reinterpret_tensor(buf528, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg344_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf537)
    del arg344_1
    del arg345_1
    buf538 = buf535; del buf535  # reuse
    buf539 = buf531; del buf531  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_150(c_void_p(buf538.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf539.data_ptr()))
    buf540 = reinterpret_tensor(buf537, (24, 512, 64), (32768, 64, 1), 0); del buf537  # reuse
    # Source Nodes: [context_layer_63], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf538, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf539, (24, 512, 64), (32768, 64, 1), 0), out=buf540)
    buf541 = reinterpret_tensor(buf539, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf539  # reuse
    cpp_fused_clone_151(c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()))
    buf542 = reinterpret_tensor(buf540, (512, 1536), (1536, 1), 0); del buf540  # reuse
    # Source Nodes: [hidden_states_168], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg347_1, reinterpret_tensor(buf541, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg346_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf542)
    del arg346_1
    del arg347_1
    buf543 = buf526; del buf526  # reuse
    buf544 = buf525; del buf525  # reuse
    buf546 = reinterpret_tensor(buf541, (1, 512, 1536), (786432, 1536, 1), 0); del buf541  # reuse
    cpp_fused_add_native_layer_norm_152(c_void_p(buf542.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf546.data_ptr()))
    del arg348_1
    del arg349_1
    buf547 = reinterpret_tensor(buf523, (512, 6144), (6144, 1), 0); del buf523  # reuse
    # Source Nodes: [hidden_states_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg351_1, reinterpret_tensor(buf546, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg350_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf547)
    del arg350_1
    del arg351_1
    buf548 = reinterpret_tensor(buf547, (1, 512, 6144), (3145728, 6144, 1), 0); del buf547  # reuse
    cpp_fused_gelu_153(c_void_p(buf548.data_ptr()))
    buf549 = buf542; del buf542  # reuse
    # Source Nodes: [hidden_states_173], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg353_1, reinterpret_tensor(buf548, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg352_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf549)
    del arg352_1
    del arg353_1
    buf550 = buf544; del buf544  # reuse
    buf551 = buf543; del buf543  # reuse
    buf553 = buf528; del buf528  # reuse
    cpp_fused_add_native_layer_norm_154(c_void_p(buf549.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf553.data_ptr()))
    del arg354_1
    del arg355_1
    buf554 = buf549; del buf549  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_22_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg357_1, reinterpret_tensor(buf553, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg356_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf554)
    del arg356_1
    del arg357_1
    buf555 = reinterpret_tensor(buf546, (512, 1536), (1536, 1), 0); del buf546  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_22_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg359_1, reinterpret_tensor(buf553, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg358_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf555)
    del arg358_1
    del arg359_1
    buf556 = reinterpret_tensor(buf529, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf529  # reuse
    buf557 = reinterpret_tensor(buf555, (24, 64, 512), (64, 1, 1536), 0); del buf555  # reuse
    cpp_fused_clone_div_sqrt_155(c_void_p(buf557.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf556.data_ptr()))
    buf558 = reinterpret_tensor(buf538, (24, 512, 512), (262144, 512, 1), 0); del buf538  # reuse
    # Source Nodes: [attention_scores_66, scale_22, truediv_22], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf556, (24, 512, 64), (32768, 64, 1), 0), buf557, out=buf558)
    buf559 = buf536; del buf536  # reuse
    buf560 = reinterpret_tensor(buf558, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf558  # reuse
    buf561 = buf534; del buf534  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_156(c_void_p(buf560.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf561.data_ptr()))
    buf562 = reinterpret_tensor(buf557, (512, 1536), (1536, 1), 0); del buf557  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_22_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg361_1, reinterpret_tensor(buf553, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg360_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf562)
    del arg360_1
    del arg361_1
    buf563 = buf560; del buf560  # reuse
    buf564 = buf556; del buf556  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_157(c_void_p(buf563.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf564.data_ptr()))
    buf565 = reinterpret_tensor(buf562, (24, 512, 64), (32768, 64, 1), 0); del buf562  # reuse
    # Source Nodes: [context_layer_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf563, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf564, (24, 512, 64), (32768, 64, 1), 0), out=buf565)
    buf566 = reinterpret_tensor(buf564, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf564  # reuse
    cpp_fused_clone_158(c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()))
    buf567 = reinterpret_tensor(buf565, (512, 1536), (1536, 1), 0); del buf565  # reuse
    # Source Nodes: [hidden_states_176], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg363_1, reinterpret_tensor(buf566, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg362_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf567)
    del arg362_1
    del arg363_1
    buf568 = buf551; del buf551  # reuse
    buf569 = buf550; del buf550  # reuse
    buf571 = reinterpret_tensor(buf566, (1, 512, 1536), (786432, 1536, 1), 0); del buf566  # reuse
    cpp_fused_add_native_layer_norm_159(c_void_p(buf567.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf571.data_ptr()))
    del arg364_1
    del arg365_1
    buf572 = reinterpret_tensor(buf548, (512, 6144), (6144, 1), 0); del buf548  # reuse
    # Source Nodes: [hidden_states_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg367_1, reinterpret_tensor(buf571, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg366_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf572)
    del arg366_1
    del arg367_1
    buf573 = reinterpret_tensor(buf572, (1, 512, 6144), (3145728, 6144, 1), 0); del buf572  # reuse
    cpp_fused_gelu_160(c_void_p(buf573.data_ptr()))
    buf574 = buf567; del buf567  # reuse
    # Source Nodes: [hidden_states_181], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg369_1, reinterpret_tensor(buf573, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg368_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf574)
    del arg368_1
    del arg369_1
    buf575 = buf569; del buf569  # reuse
    buf576 = buf568; del buf568  # reuse
    buf578 = buf553; del buf553  # reuse
    cpp_fused_add_native_layer_norm_161(c_void_p(buf574.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf578.data_ptr()))
    del arg370_1
    del arg371_1
    buf579 = buf574; del buf574  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_23_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg373_1, reinterpret_tensor(buf578, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg372_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf579)
    del arg372_1
    del arg373_1
    buf580 = reinterpret_tensor(buf571, (512, 1536), (1536, 1), 0); del buf571  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_23_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg375_1, reinterpret_tensor(buf578, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg374_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf580)
    del arg374_1
    del arg375_1
    buf581 = reinterpret_tensor(buf554, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf554  # reuse
    buf582 = reinterpret_tensor(buf580, (24, 64, 512), (64, 1, 1536), 0); del buf580  # reuse
    cpp_fused_clone_div_sqrt_162(c_void_p(buf582.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf581.data_ptr()))
    del buf579
    buf583 = reinterpret_tensor(buf563, (24, 512, 512), (262144, 512, 1), 0); del buf563  # reuse
    # Source Nodes: [attention_scores_69, scale_23, truediv_23], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf581, (24, 512, 64), (32768, 64, 1), 0), buf582, out=buf583)
    buf584 = buf561; del buf561  # reuse
    buf585 = reinterpret_tensor(buf583, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf583  # reuse
    buf586 = buf559; del buf559  # reuse
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_163(c_void_p(buf585.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf586.data_ptr()))
    del buf584
    buf587 = reinterpret_tensor(buf582, (512, 1536), (1536, 1), 0); del buf582  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_23_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg377_1, reinterpret_tensor(buf578, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg376_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf587)
    del arg376_1
    del arg377_1
    buf588 = buf585; del buf585  # reuse
    buf589 = buf581; del buf581  # reuse
    cpp_fused__softmax_bitwise_not_clone_masked_fill_164(c_void_p(buf588.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf589.data_ptr()))
    del buf586
    buf590 = reinterpret_tensor(buf587, (24, 512, 64), (32768, 64, 1), 0); del buf587  # reuse
    # Source Nodes: [context_layer_69], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf588, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf589, (24, 512, 64), (32768, 64, 1), 0), out=buf590)
    del buf588
    buf591 = reinterpret_tensor(buf589, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf589  # reuse
    cpp_fused_clone_165(c_void_p(buf590.data_ptr()), c_void_p(buf591.data_ptr()))
    buf592 = reinterpret_tensor(buf590, (512, 1536), (1536, 1), 0); del buf590  # reuse
    # Source Nodes: [hidden_states_184], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg379_1, reinterpret_tensor(buf591, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg378_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf592)
    del arg378_1
    del arg379_1
    buf593 = buf576; del buf576  # reuse
    buf594 = buf575; del buf575  # reuse
    buf596 = reinterpret_tensor(buf591, (1, 512, 1536), (786432, 1536, 1), 0); del buf591  # reuse
    cpp_fused_add_native_layer_norm_166(c_void_p(buf592.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg381_1.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf596.data_ptr()))
    del arg380_1
    del arg381_1
    buf597 = reinterpret_tensor(buf573, (512, 6144), (6144, 1), 0); del buf573  # reuse
    # Source Nodes: [hidden_states_186], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg383_1, reinterpret_tensor(buf596, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg382_1, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf597)
    del arg382_1
    del arg383_1
    buf598 = reinterpret_tensor(buf597, (1, 512, 6144), (3145728, 6144, 1), 0); del buf597  # reuse
    cpp_fused_gelu_167(c_void_p(buf598.data_ptr()))
    buf599 = buf592; del buf592  # reuse
    # Source Nodes: [hidden_states_189], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg385_1, reinterpret_tensor(buf598, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg384_1, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf599)
    del arg384_1
    del arg385_1
    del buf598
    buf600 = buf594; del buf594  # reuse
    buf601 = buf593; del buf593  # reuse
    buf603 = buf578; del buf578  # reuse
    cpp_fused_add_native_layer_norm_168(c_void_p(buf599.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg387_1.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf603.data_ptr()))
    del arg386_1
    del arg387_1
    del buf596
    del buf599
    buf604 = empty((512, 2), device='cpu', dtype=torch.float32)
    # Source Nodes: [logits], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg389_1, reinterpret_tensor(buf603, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg388_1, (1536, 2), (1, 1536), 0), alpha=1, beta=1, out=buf604)
    del arg388_1
    del arg389_1
    del buf603
    buf605 = reinterpret_tensor(buf601, (1, 512), (512, 1), 0); del buf601  # reuse
    buf606 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf607 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf608 = reinterpret_tensor(buf600, (1, 512), (512, 1), 0); del buf600  # reuse
    buf609 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf610 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf611 = reinterpret_tensor(buf606, (), (), 0); del buf606  # reuse
    buf612 = buf611; del buf611  # reuse
    cpp_fused__log_softmax_add_clamp_clone_div_nll_loss_forward_169(c_void_p(buf612.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(arg392_1.data_ptr()), c_void_p(arg393_1.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf610.data_ptr()))
    del arg392_1
    del arg393_1
    return (buf612, buf605, buf608, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128100, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((512, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((2, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg391_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg392_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    arg393_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DebertaV2ForQuestionAnswering', benchmark_compiled_module)
