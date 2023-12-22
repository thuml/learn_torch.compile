
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 50257);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 50257L), "index out of bounds: 0 <= tmp3 < 50257L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 50257);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 50257L), "index out of bounds: 0 <= tmp3 < 50257L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_full_where_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_2 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_native_layer_norm_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const long* in_ptr1,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = decltype(tmp1)(tmp1 + 50257);
                        auto tmp3 = tmp1 < 0;
                        auto tmp4 = tmp3 ? tmp2 : tmp1;
                        TORCH_CHECK((0 <= tmp4) & (tmp4 < 50257L), "index out of bounds: 0 <= tmp4 < 50257L")
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*tmp4)));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp0 + tmp7;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = decltype(tmp1)(tmp1 + 50257);
                    auto tmp3 = tmp1 < 0;
                    auto tmp4 = tmp3 ? tmp2 : tmp1;
                    TORCH_CHECK((0 <= tmp4) & (tmp4 < 50257L), "index out of bounds: 0 <= tmp4 < 50257L")
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*tmp4)));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp0 + tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp13 = static_cast<float>(768.0);
                    auto tmp14 = tmp12 / tmp13;
                    auto tmp15 = static_cast<float>(1e-05);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    auto tmp17 = 1 / std::sqrt(tmp16);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp21 = tmp19 * tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    tmp23.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_embedding_native_layer_norm_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = decltype(tmp1)(tmp1 + 50257);
                        auto tmp3 = tmp1 < 0;
                        auto tmp4 = tmp3 ? tmp2 : tmp1;
                        TORCH_CHECK((0 <= tmp4) & (tmp4 < 50257L), "index out of bounds: 0 <= tmp4 < 50257L")
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*tmp4)));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp0 + tmp7;
                        auto tmp10 = tmp8 + tmp9;
                        tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_full_where_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_7 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_tanh_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_10 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_full_where_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_12 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_full_where_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_17 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_tanh_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_full_where_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_22 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_full_where_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_27 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_tanh_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_30 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_nll_loss_forward_31 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50257L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(50256L); x1<static_cast<long>(50257L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (50257L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50257L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(50256L); x1<static_cast<long>(50257L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (50257L*x0))];
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
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(511L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(1L + x0)];
                        auto tmp9 = out_ptr0[static_cast<long>(x0)];
                        auto tmp11 = out_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = static_cast<long>(0);
                        auto tmp4 = tmp2 ? tmp0 : tmp3;
                        auto tmp5 = decltype(tmp4)(tmp4 + 50257);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 50257L), "index out of bounds: 0 <= tmp7 < 50257L")
                        auto tmp8 = in_ptr0[static_cast<long>(tmp7 + (50257L*x0))];
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2304, ), (1, ))
    assert_size_stride(arg1_1, (768, 2304), (2304, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, 768), (768, 1))
    assert_size_stride(arg4_1, (3072, ), (1, ))
    assert_size_stride(arg5_1, (768, 3072), (3072, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (3072, 768), (768, 1))
    assert_size_stride(arg8_1, (2304, ), (1, ))
    assert_size_stride(arg9_1, (768, 2304), (2304, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (3072, ), (1, ))
    assert_size_stride(arg13_1, (768, 3072), (3072, 1))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (3072, 768), (768, 1))
    assert_size_stride(arg16_1, (2304, ), (1, ))
    assert_size_stride(arg17_1, (768, 2304), (2304, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, 768), (768, 1))
    assert_size_stride(arg20_1, (3072, ), (1, ))
    assert_size_stride(arg21_1, (768, 3072), (3072, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (3072, 768), (768, 1))
    assert_size_stride(arg24_1, (2304, ), (1, ))
    assert_size_stride(arg25_1, (768, 2304), (2304, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (3072, ), (1, ))
    assert_size_stride(arg29_1, (768, 3072), (3072, 1))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (3072, 768), (768, 1))
    assert_size_stride(arg32_1, (2304, ), (1, ))
    assert_size_stride(arg33_1, (768, 2304), (2304, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, 768), (768, 1))
    assert_size_stride(arg36_1, (3072, ), (1, ))
    assert_size_stride(arg37_1, (768, 3072), (3072, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (3072, 768), (768, 1))
    assert_size_stride(arg40_1, (2304, ), (1, ))
    assert_size_stride(arg41_1, (768, 2304), (2304, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (3072, ), (1, ))
    assert_size_stride(arg45_1, (768, 3072), (3072, 1))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (3072, 768), (768, 1))
    assert_size_stride(arg48_1, (50257, 768), (768, 1))
    assert_size_stride(arg49_1, (1024, 768), (768, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (50257, 768), (768, 1))
    assert_size_stride(arg77_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg78_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg79_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg80_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg81_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg82_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg83_1, (1, 512), (512, 1))
    assert_size_stride(arg84_1, (1, 512), (512, 1))
    buf0 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_0(c_void_p(arg83_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg50_1
    del arg51_1
    buf4 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg0_1, reinterpret_tensor(buf3, (512, 768), (768, 1), 0), arg1_1, alpha=1, beta=1, out=buf4)
    del arg0_1
    del arg1_1
    buf5 = empty((12, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf4, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf4, (12, 64, 512), (64, 1, 2304), 768), out=buf5)
    buf6 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf7 = reinterpret_tensor(buf5, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf5  # reuse
    buf8 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf9 = buf7; del buf7  # reuse
    cpp_fused__softmax_div_full_where_1(c_void_p(buf9.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()))
    del arg77_1
    buf10 = reinterpret_tensor(buf3, (12, 512, 64), (32768, 64, 1), 0); del buf3  # reuse
    # Source Nodes: [attn_output], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf9, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf4, (12, 512, 64), (64, 2304, 1), 1536), out=buf10)
    buf11 = empty((1, 512, 12, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_2(c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    buf12 = reinterpret_tensor(buf10, (512, 768), (768, 1), 0); del buf10  # reuse
    # Source Nodes: [x_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg2_1, reinterpret_tensor(buf11, (512, 768), (768, 1), 0), arg3_1, alpha=1, beta=1, out=buf12)
    del arg2_1
    del arg3_1
    buf13 = buf1; del buf1  # reuse
    buf14 = buf0; del buf0  # reuse
    buf16 = reinterpret_tensor(buf11, (1, 512, 768), (393216, 768, 1), 0); del buf11  # reuse
    cpp_fused_add_embedding_native_layer_norm_3(c_void_p(buf12.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf16.data_ptr()))
    del arg52_1
    del arg53_1
    buf17 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg4_1, reinterpret_tensor(buf16, (512, 768), (768, 1), 0), arg5_1, alpha=1, beta=1, out=buf17)
    del arg4_1
    del arg5_1
    buf18 = reinterpret_tensor(buf17, (1, 512, 3072), (1572864, 3072, 1), 0); del buf17  # reuse
    cpp_fused_add_mul_pow_tanh_4(c_void_p(buf18.data_ptr()))
    buf19 = reinterpret_tensor(buf16, (512, 768), (768, 1), 0); del buf16  # reuse
    # Source Nodes: [x_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg6_1, reinterpret_tensor(buf18, (512, 3072), (3072, 1), 0), arg7_1, alpha=1, beta=1, out=buf19)
    del arg6_1
    del arg7_1
    buf20 = reinterpret_tensor(buf19, (1, 512, 768), (393216, 768, 1), 0); del buf19  # reuse
    buf21 = buf14; del buf14  # reuse
    buf22 = buf13; del buf13  # reuse
    buf24 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_5(c_void_p(buf20.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()))
    del arg48_1
    del arg49_1
    del arg54_1
    del arg55_1
    del arg83_1
    buf25 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf24, (512, 768), (768, 1), 0), arg9_1, alpha=1, beta=1, out=buf25)
    del arg8_1
    del arg9_1
    buf26 = reinterpret_tensor(buf9, (12, 512, 512), (262144, 512, 1), 0); del buf9  # reuse
    # Source Nodes: [attn_weights_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf25, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf25, (12, 64, 512), (64, 1, 2304), 768), out=buf26)
    buf27 = buf8; del buf8  # reuse
    buf28 = reinterpret_tensor(buf26, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf26  # reuse
    buf29 = buf6; del buf6  # reuse
    buf30 = buf28; del buf28  # reuse
    cpp_fused__softmax_div_full_where_6(c_void_p(buf30.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf29.data_ptr()))
    del arg78_1
    buf31 = reinterpret_tensor(buf24, (12, 512, 64), (32768, 64, 1), 0); del buf24  # reuse
    # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf30, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf25, (12, 512, 64), (64, 2304, 1), 1536), out=buf31)
    buf32 = reinterpret_tensor(buf12, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf12  # reuse
    cpp_fused_clone_7(c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()))
    buf33 = reinterpret_tensor(buf31, (512, 768), (768, 1), 0); del buf31  # reuse
    # Source Nodes: [x_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf32, (512, 768), (768, 1), 0), arg11_1, alpha=1, beta=1, out=buf33)
    del arg10_1
    del arg11_1
    buf34 = buf22; del buf22  # reuse
    buf35 = buf21; del buf21  # reuse
    buf37 = reinterpret_tensor(buf32, (1, 512, 768), (393216, 768, 1), 0); del buf32  # reuse
    cpp_fused_add_native_layer_norm_8(c_void_p(buf33.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf37.data_ptr()))
    del arg56_1
    del arg57_1
    buf38 = reinterpret_tensor(buf18, (512, 3072), (3072, 1), 0); del buf18  # reuse
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf37, (512, 768), (768, 1), 0), arg13_1, alpha=1, beta=1, out=buf38)
    del arg12_1
    del arg13_1
    buf39 = reinterpret_tensor(buf38, (1, 512, 3072), (1572864, 3072, 1), 0); del buf38  # reuse
    cpp_fused_add_mul_pow_tanh_9(c_void_p(buf39.data_ptr()))
    buf40 = reinterpret_tensor(buf37, (512, 768), (768, 1), 0); del buf37  # reuse
    # Source Nodes: [x_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf39, (512, 3072), (3072, 1), 0), arg15_1, alpha=1, beta=1, out=buf40)
    del arg14_1
    del arg15_1
    buf41 = buf35; del buf35  # reuse
    buf42 = buf34; del buf34  # reuse
    buf44 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_10(c_void_p(buf33.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf44.data_ptr()))
    del arg58_1
    del arg59_1
    buf45 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg16_1, reinterpret_tensor(buf44, (512, 768), (768, 1), 0), arg17_1, alpha=1, beta=1, out=buf45)
    del arg16_1
    del arg17_1
    buf46 = reinterpret_tensor(buf30, (12, 512, 512), (262144, 512, 1), 0); del buf30  # reuse
    # Source Nodes: [attn_weights_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf45, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf45, (12, 64, 512), (64, 1, 2304), 768), out=buf46)
    buf47 = buf29; del buf29  # reuse
    buf48 = reinterpret_tensor(buf46, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf46  # reuse
    buf49 = buf27; del buf27  # reuse
    buf50 = buf48; del buf48  # reuse
    cpp_fused__softmax_div_full_where_11(c_void_p(buf50.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()))
    del arg79_1
    buf51 = reinterpret_tensor(buf44, (12, 512, 64), (32768, 64, 1), 0); del buf44  # reuse
    # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf50, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf45, (12, 512, 64), (64, 2304, 1), 1536), out=buf51)
    buf52 = empty((1, 512, 12, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_12(c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    buf53 = reinterpret_tensor(buf51, (512, 768), (768, 1), 0); del buf51  # reuse
    # Source Nodes: [x_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf52, (512, 768), (768, 1), 0), arg19_1, alpha=1, beta=1, out=buf53)
    del arg18_1
    del arg19_1
    buf54 = buf42; del buf42  # reuse
    buf55 = buf41; del buf41  # reuse
    buf57 = reinterpret_tensor(buf52, (1, 512, 768), (393216, 768, 1), 0); del buf52  # reuse
    cpp_fused_add_native_layer_norm_13(c_void_p(buf53.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg60_1
    del arg61_1
    buf58 = reinterpret_tensor(buf39, (512, 3072), (3072, 1), 0); del buf39  # reuse
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf57, (512, 768), (768, 1), 0), arg21_1, alpha=1, beta=1, out=buf58)
    del arg20_1
    del arg21_1
    buf59 = reinterpret_tensor(buf58, (1, 512, 3072), (1572864, 3072, 1), 0); del buf58  # reuse
    cpp_fused_add_mul_pow_tanh_14(c_void_p(buf59.data_ptr()))
    buf60 = reinterpret_tensor(buf57, (512, 768), (768, 1), 0); del buf57  # reuse
    # Source Nodes: [x_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg22_1, reinterpret_tensor(buf59, (512, 3072), (3072, 1), 0), arg23_1, alpha=1, beta=1, out=buf60)
    del arg22_1
    del arg23_1
    buf61 = reinterpret_tensor(buf60, (1, 512, 768), (393216, 768, 1), 0); del buf60  # reuse
    buf62 = buf55; del buf55  # reuse
    buf63 = buf54; del buf54  # reuse
    buf65 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_15(c_void_p(buf61.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf65.data_ptr()))
    del arg62_1
    del arg63_1
    buf66 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg24_1, reinterpret_tensor(buf65, (512, 768), (768, 1), 0), arg25_1, alpha=1, beta=1, out=buf66)
    del arg24_1
    del arg25_1
    buf67 = reinterpret_tensor(buf50, (12, 512, 512), (262144, 512, 1), 0); del buf50  # reuse
    # Source Nodes: [attn_weights_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf66, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf66, (12, 64, 512), (64, 1, 2304), 768), out=buf67)
    buf68 = buf49; del buf49  # reuse
    buf69 = reinterpret_tensor(buf67, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf67  # reuse
    buf70 = buf47; del buf47  # reuse
    buf71 = buf69; del buf69  # reuse
    cpp_fused__softmax_div_full_where_16(c_void_p(buf71.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf70.data_ptr()))
    del arg80_1
    buf72 = reinterpret_tensor(buf65, (12, 512, 64), (32768, 64, 1), 0); del buf65  # reuse
    # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf71, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf66, (12, 512, 64), (64, 2304, 1), 1536), out=buf72)
    buf73 = reinterpret_tensor(buf53, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf53  # reuse
    cpp_fused_clone_17(c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    buf74 = reinterpret_tensor(buf72, (512, 768), (768, 1), 0); del buf72  # reuse
    # Source Nodes: [x_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg26_1, reinterpret_tensor(buf73, (512, 768), (768, 1), 0), arg27_1, alpha=1, beta=1, out=buf74)
    del arg26_1
    del arg27_1
    buf75 = buf63; del buf63  # reuse
    buf76 = buf62; del buf62  # reuse
    buf78 = reinterpret_tensor(buf73, (1, 512, 768), (393216, 768, 1), 0); del buf73  # reuse
    cpp_fused_add_native_layer_norm_18(c_void_p(buf74.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()))
    del arg64_1
    del arg65_1
    buf79 = reinterpret_tensor(buf59, (512, 3072), (3072, 1), 0); del buf59  # reuse
    # Source Nodes: [x_28], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg28_1, reinterpret_tensor(buf78, (512, 768), (768, 1), 0), arg29_1, alpha=1, beta=1, out=buf79)
    del arg28_1
    del arg29_1
    buf80 = reinterpret_tensor(buf79, (1, 512, 3072), (1572864, 3072, 1), 0); del buf79  # reuse
    cpp_fused_add_mul_pow_tanh_19(c_void_p(buf80.data_ptr()))
    buf81 = reinterpret_tensor(buf78, (512, 768), (768, 1), 0); del buf78  # reuse
    # Source Nodes: [x_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg30_1, reinterpret_tensor(buf80, (512, 3072), (3072, 1), 0), arg31_1, alpha=1, beta=1, out=buf81)
    del arg30_1
    del arg31_1
    buf82 = buf76; del buf76  # reuse
    buf83 = buf75; del buf75  # reuse
    buf85 = reinterpret_tensor(buf40, (1, 512, 768), (393216, 768, 1), 0); del buf40  # reuse
    cpp_fused_add_native_layer_norm_20(c_void_p(buf74.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf85.data_ptr()))
    del arg66_1
    del arg67_1
    buf86 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg32_1, reinterpret_tensor(buf85, (512, 768), (768, 1), 0), arg33_1, alpha=1, beta=1, out=buf86)
    del arg32_1
    del arg33_1
    buf87 = reinterpret_tensor(buf71, (12, 512, 512), (262144, 512, 1), 0); del buf71  # reuse
    # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf86, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf86, (12, 64, 512), (64, 1, 2304), 768), out=buf87)
    buf88 = buf70; del buf70  # reuse
    buf89 = reinterpret_tensor(buf87, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf87  # reuse
    buf90 = buf68; del buf68  # reuse
    buf91 = buf89; del buf89  # reuse
    cpp_fused__softmax_div_full_where_21(c_void_p(buf91.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()))
    del arg81_1
    buf92 = reinterpret_tensor(buf85, (12, 512, 64), (32768, 64, 1), 0); del buf85  # reuse
    # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf91, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf86, (12, 512, 64), (64, 2304, 1), 1536), out=buf92)
    buf93 = reinterpret_tensor(buf33, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf33  # reuse
    cpp_fused_clone_22(c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()))
    buf94 = reinterpret_tensor(buf92, (512, 768), (768, 1), 0); del buf92  # reuse
    # Source Nodes: [x_34], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg34_1, reinterpret_tensor(buf93, (512, 768), (768, 1), 0), arg35_1, alpha=1, beta=1, out=buf94)
    del arg34_1
    del arg35_1
    buf95 = buf83; del buf83  # reuse
    buf96 = buf82; del buf82  # reuse
    buf98 = reinterpret_tensor(buf93, (1, 512, 768), (393216, 768, 1), 0); del buf93  # reuse
    cpp_fused_add_native_layer_norm_23(c_void_p(buf94.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf98.data_ptr()))
    del arg68_1
    del arg69_1
    buf99 = reinterpret_tensor(buf80, (512, 3072), (3072, 1), 0); del buf80  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg36_1, reinterpret_tensor(buf98, (512, 768), (768, 1), 0), arg37_1, alpha=1, beta=1, out=buf99)
    del arg36_1
    del arg37_1
    buf100 = reinterpret_tensor(buf99, (1, 512, 3072), (1572864, 3072, 1), 0); del buf99  # reuse
    cpp_fused_add_mul_pow_tanh_24(c_void_p(buf100.data_ptr()))
    buf101 = reinterpret_tensor(buf98, (512, 768), (768, 1), 0); del buf98  # reuse
    # Source Nodes: [x_38], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg38_1, reinterpret_tensor(buf100, (512, 3072), (3072, 1), 0), arg39_1, alpha=1, beta=1, out=buf101)
    del arg38_1
    del arg39_1
    buf102 = reinterpret_tensor(buf101, (1, 512, 768), (393216, 768, 1), 0); del buf101  # reuse
    buf103 = buf96; del buf96  # reuse
    buf104 = buf95; del buf95  # reuse
    buf106 = buf20; del buf20  # reuse
    cpp_fused_add_native_layer_norm_25(c_void_p(buf102.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()))
    del arg70_1
    del arg71_1
    del buf61
    del buf74
    buf107 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg40_1, reinterpret_tensor(buf106, (512, 768), (768, 1), 0), arg41_1, alpha=1, beta=1, out=buf107)
    del arg40_1
    del arg41_1
    buf108 = reinterpret_tensor(buf91, (12, 512, 512), (262144, 512, 1), 0); del buf91  # reuse
    # Source Nodes: [attn_weights_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf107, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf107, (12, 64, 512), (64, 1, 2304), 768), out=buf108)
    buf109 = buf90; del buf90  # reuse
    buf110 = reinterpret_tensor(buf108, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf108  # reuse
    buf111 = buf88; del buf88  # reuse
    buf112 = buf110; del buf110  # reuse
    cpp_fused__softmax_div_full_where_26(c_void_p(buf112.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()))
    del arg82_1
    del buf109
    del buf111
    buf113 = reinterpret_tensor(buf106, (12, 512, 64), (32768, 64, 1), 0); del buf106  # reuse
    # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf112, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf107, (12, 512, 64), (64, 2304, 1), 1536), out=buf113)
    del buf112
    buf114 = reinterpret_tensor(buf94, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf94  # reuse
    cpp_fused_clone_27(c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()))
    buf115 = reinterpret_tensor(buf113, (512, 768), (768, 1), 0); del buf113  # reuse
    # Source Nodes: [x_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg42_1, reinterpret_tensor(buf114, (512, 768), (768, 1), 0), arg43_1, alpha=1, beta=1, out=buf115)
    del arg42_1
    del arg43_1
    buf116 = buf104; del buf104  # reuse
    buf117 = buf103; del buf103  # reuse
    buf119 = reinterpret_tensor(buf114, (1, 512, 768), (393216, 768, 1), 0); del buf114  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf115.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()))
    del arg72_1
    del arg73_1
    buf120 = reinterpret_tensor(buf100, (512, 3072), (3072, 1), 0); del buf100  # reuse
    # Source Nodes: [x_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg44_1, reinterpret_tensor(buf119, (512, 768), (768, 1), 0), arg45_1, alpha=1, beta=1, out=buf120)
    del arg44_1
    del arg45_1
    buf121 = reinterpret_tensor(buf120, (1, 512, 3072), (1572864, 3072, 1), 0); del buf120  # reuse
    cpp_fused_add_mul_pow_tanh_29(c_void_p(buf121.data_ptr()))
    buf122 = reinterpret_tensor(buf119, (512, 768), (768, 1), 0); del buf119  # reuse
    # Source Nodes: [x_46], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg46_1, reinterpret_tensor(buf121, (512, 3072), (3072, 1), 0), arg47_1, alpha=1, beta=1, out=buf122)
    del arg46_1
    del arg47_1
    del buf121
    buf123 = buf117; del buf117  # reuse
    buf124 = buf116; del buf116  # reuse
    buf126 = reinterpret_tensor(buf81, (1, 512, 768), (393216, 768, 1), 0); del buf81  # reuse
    cpp_fused_add_native_layer_norm_30(c_void_p(buf115.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()))
    del arg74_1
    del arg75_1
    del buf102
    del buf115
    del buf122
    del buf123
    del buf124
    buf127 = empty((512, 50257), device='cpu', dtype=torch.float32)
    # Source Nodes: [lm_logits], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf126, (512, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 50257), (1, 768), 0), out=buf127)
    del arg76_1
    del buf126
    buf128 = empty_strided((511, 1), (1, 511), device='cpu', dtype=torch.float32)
    buf129 = empty_strided((511, 1), (1, 511), device='cpu', dtype=torch.float32)
    buf130 = empty((), device='cpu', dtype=torch.float32)
    buf131 = empty((), device='cpu', dtype=torch.int64)
    buf132 = buf130; del buf130  # reuse
    cpp_fused__log_softmax_nll_loss_forward_31(c_void_p(buf132.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf131.data_ptr()))
    del arg84_1
    return (buf132, reinterpret_tensor(buf127, (1, 512, 50257), (25731584, 50257, 1), 0), reinterpret_tensor(buf4, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf4, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf25, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf25, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf45, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf45, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf66, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf66, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf86, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf86, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf107, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf107, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((50257, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((50257, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg78_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg79_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg80_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg81_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg82_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg83_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg84_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistillGPT2', benchmark_compiled_module)
