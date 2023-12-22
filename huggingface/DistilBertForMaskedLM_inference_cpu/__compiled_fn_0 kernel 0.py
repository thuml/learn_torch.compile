
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
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = in_ptr2[static_cast<long>(x0)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 30522);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 30522L), "index out of bounds: 0 <= tmp3 < 30522L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                    auto tmp6 = decltype(tmp5)(tmp5 + 512);
                    auto tmp7 = tmp5 < 0;
                    auto tmp8 = tmp7 ? tmp6 : tmp5;
                    TORCH_CHECK((0 <= tmp8) & (tmp8 < 512L), "index out of bounds: 0 <= tmp8 < 512L")
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*tmp8)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp5 = in_ptr2[static_cast<long>(x0)];
                auto tmp11 = out_ptr0[static_cast<long>(x0)];
                auto tmp14 = out_ptr1[static_cast<long>(x0)];
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = decltype(tmp0)(tmp0 + 30522);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 30522L), "index out of bounds: 0 <= tmp3 < 30522L")
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                auto tmp6 = decltype(tmp5)(tmp5 + 512);
                auto tmp7 = tmp5 < 0;
                auto tmp8 = tmp7 ? tmp6 : tmp5;
                TORCH_CHECK((0 <= tmp8) & (tmp8 < 512L), "index out of bounds: 0 <= tmp8 < 512L")
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*tmp8)));
                auto tmp10 = tmp4 + tmp9;
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp10 - tmp12;
                auto tmp15 = static_cast<float>(768.0);
                auto tmp16 = tmp14 / tmp15;
                auto tmp17 = static_cast<float>(1e-12);
                auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                auto tmp19 = 1 / std::sqrt(tmp18);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp13 * tmp20;
                auto tmp23 = tmp21 * tmp22;
                auto tmp25 = tmp23 + tmp24;
                tmp25.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_div_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_masked_fill_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp0 = static_cast<float>(1.0);
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x0)];
                        auto tmp0 = static_cast<float>(1.0);
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_gelu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_div_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_masked_fill_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp0 = static_cast<float>(1.0);
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x0)];
                        auto tmp0 = static_cast<float>(1.0);
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_gelu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_div_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_masked_fill_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp0 = static_cast<float>(1.0);
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x0)];
                        auto tmp0 = static_cast<float>(1.0);
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_gelu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_div_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_masked_fill_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp0 = static_cast<float>(1.0);
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x0)];
                        auto tmp0 = static_cast<float>(1.0);
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_gelu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_div_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_masked_fill_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp0 = static_cast<float>(1.0);
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x0)];
                        auto tmp0 = static_cast<float>(1.0);
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_gelu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_div_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_masked_fill_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp0 = static_cast<float>(1.0);
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x0)];
                        auto tmp0 = static_cast<float>(1.0);
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_gelu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_gelu_native_layer_norm_43 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__log_softmax_nll_loss_forward_44 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(30520L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (30522L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(30520L); x1<static_cast<long>(30522L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (30522L*x0))];
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(30520L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (30522L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(30520L); x1<static_cast<long>(30522L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (30522L*x0))];
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
                        auto tmp5 = decltype(tmp4)(tmp4 + 30522);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 30522L), "index out of bounds: 0 <= tmp7 < 30522L")
                        auto tmp8 = in_ptr0[static_cast<long>(tmp7 + (30522L*x0))];
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1 = args
    args.clear()
    assert_size_stride(arg0_1, (30522, 768), (768, 1))
    assert_size_stride(arg1_1, (512, 768), (768, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, 768), (768, 1))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, 768), (768, 1))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, 768), (768, 1))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, 768), (768, 1))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (3072, 768), (768, 1))
    assert_size_stride(arg15_1, (3072, ), (1, ))
    assert_size_stride(arg16_1, (768, 3072), (3072, 1))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, 768), (768, 1))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, 768), (768, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, 768), (768, 1))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, 768), (768, 1))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (3072, 768), (768, 1))
    assert_size_stride(arg31_1, (3072, ), (1, ))
    assert_size_stride(arg32_1, (768, 3072), (3072, 1))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, 768), (768, 1))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, 768), (768, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, 768), (768, 1))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, 768), (768, 1))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (3072, 768), (768, 1))
    assert_size_stride(arg47_1, (3072, ), (1, ))
    assert_size_stride(arg48_1, (768, 3072), (3072, 1))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, 768), (768, 1))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, 768), (768, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, 768), (768, 1))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (3072, 768), (768, 1))
    assert_size_stride(arg63_1, (3072, ), (1, ))
    assert_size_stride(arg64_1, (768, 3072), (3072, 1))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, 768), (768, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, 768), (768, 1))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, 768), (768, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, 768), (768, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (3072, 768), (768, 1))
    assert_size_stride(arg79_1, (3072, ), (1, ))
    assert_size_stride(arg80_1, (768, 3072), (3072, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, 768), (768, 1))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, 768), (768, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, 768), (768, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, 768), (768, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (3072, 768), (768, 1))
    assert_size_stride(arg95_1, (3072, ), (1, ))
    assert_size_stride(arg96_1, (768, 3072), (3072, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, 768), (768, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (30522, 768), (768, 1))
    assert_size_stride(arg105_1, (30522, ), (1, ))
    assert_size_stride(arg106_1, (1, 512), (512, 1))
    assert_size_stride(arg107_1, (1, 128), (128, 1))
    assert_size_stride(arg108_1, (1, 128), (128, 1))
    buf0 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_0(c_void_p(arg107_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg0_1
    del arg106_1
    del arg107_1
    del arg1_1
    del arg2_1
    del arg3_1
    buf4 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_0_attention_q_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg5_1, reinterpret_tensor(buf3, (128, 768), (768, 1), 0), reinterpret_tensor(arg4_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf4)
    del arg4_1
    del arg5_1
    buf5 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_0_attention_k_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg7_1, reinterpret_tensor(buf3, (128, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf5)
    del arg6_1
    del arg7_1
    buf6 = empty((1, 12, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_div_1(c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()))
    buf7 = empty((12, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf6, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf5, (12, 64, 128), (64, 1, 768), 0), out=buf7)
    buf8 = empty_strided((1, 12, 128, 1), (1536, 128, 1, 1536), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf7, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf7  # reuse
    buf10 = empty_strided((1, 12, 128, 1), (1536, 128, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_lift_fresh_masked_fill_2(c_void_p(buf9.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf10.data_ptr()))
    buf11 = reinterpret_tensor(buf6, (128, 768), (768, 1), 0); del buf6  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_0_attention_v_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg9_1, reinterpret_tensor(buf3, (128, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf11)
    del arg8_1
    del arg9_1
    buf12 = buf9; del buf9  # reuse
    cpp_fused__softmax_3(c_void_p(buf12.data_ptr()), c_void_p(buf10.data_ptr()))
    buf13 = reinterpret_tensor(buf5, (12, 128, 64), (8192, 64, 1), 0); del buf5  # reuse
    # Source Nodes: [context], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf12, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf11, (12, 128, 64), (64, 768, 1), 0), out=buf13)
    buf14 = reinterpret_tensor(buf11, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf11  # reuse
    cpp_fused_clone_4(c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()))
    buf15 = reinterpret_tensor(buf13, (128, 768), (768, 1), 0); del buf13  # reuse
    # Source Nodes: [sa_output], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg11_1, reinterpret_tensor(buf14, (128, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf15)
    del arg10_1
    del arg11_1
    buf16 = buf1; del buf1  # reuse
    buf17 = buf0; del buf0  # reuse
    buf19 = reinterpret_tensor(buf14, (1, 128, 768), (98304, 768, 1), 0); del buf14  # reuse
    cpp_fused_add_native_layer_norm_5(c_void_p(buf15.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf19.data_ptr()))
    del arg12_1
    del arg13_1
    buf20 = empty((128, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg15_1, reinterpret_tensor(buf19, (128, 768), (768, 1), 0), reinterpret_tensor(arg14_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf20)
    del arg14_1
    del arg15_1
    buf21 = reinterpret_tensor(buf20, (1, 128, 3072), (393216, 3072, 1), 0); del buf20  # reuse
    cpp_fused_gelu_6(c_void_p(buf21.data_ptr()))
    buf22 = reinterpret_tensor(buf3, (128, 768), (768, 1), 0); del buf3  # reuse
    # Source Nodes: [x_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg17_1, reinterpret_tensor(buf21, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg16_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf22)
    del arg16_1
    del arg17_1
    buf23 = buf17; del buf17  # reuse
    buf24 = buf16; del buf16  # reuse
    buf26 = reinterpret_tensor(buf15, (1, 128, 768), (98304, 768, 1), 0); del buf15  # reuse
    cpp_fused_add_native_layer_norm_7(c_void_p(buf22.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()))
    del arg18_1
    del arg19_1
    buf27 = buf22; del buf22  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_1_attention_q_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg21_1, reinterpret_tensor(buf26, (128, 768), (768, 1), 0), reinterpret_tensor(arg20_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf27)
    del arg20_1
    del arg21_1
    buf28 = reinterpret_tensor(buf19, (128, 768), (768, 1), 0); del buf19  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_1_attention_k_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg23_1, reinterpret_tensor(buf26, (128, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf28)
    del arg22_1
    del arg23_1
    buf29 = reinterpret_tensor(buf4, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf4  # reuse
    cpp_fused_div_8(c_void_p(buf27.data_ptr()), c_void_p(buf29.data_ptr()))
    buf30 = reinterpret_tensor(buf12, (12, 128, 128), (16384, 128, 1), 0); del buf12  # reuse
    # Source Nodes: [scores_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf29, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf28, (12, 64, 128), (64, 1, 768), 0), out=buf30)
    buf31 = buf10; del buf10  # reuse
    buf32 = reinterpret_tensor(buf30, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf30  # reuse
    buf33 = buf8; del buf8  # reuse
    cpp_fused__softmax_lift_fresh_masked_fill_9(c_void_p(buf32.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()))
    buf34 = reinterpret_tensor(buf29, (128, 768), (768, 1), 0); del buf29  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_1_attention_v_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg25_1, reinterpret_tensor(buf26, (128, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf34)
    del arg24_1
    del arg25_1
    buf35 = buf32; del buf32  # reuse
    cpp_fused__softmax_10(c_void_p(buf35.data_ptr()), c_void_p(buf33.data_ptr()))
    buf36 = reinterpret_tensor(buf28, (12, 128, 64), (8192, 64, 1), 0); del buf28  # reuse
    # Source Nodes: [context_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf35, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf34, (12, 128, 64), (64, 768, 1), 0), out=buf36)
    buf37 = reinterpret_tensor(buf34, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf34  # reuse
    cpp_fused_clone_11(c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    buf38 = reinterpret_tensor(buf36, (128, 768), (768, 1), 0); del buf36  # reuse
    # Source Nodes: [sa_output_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg27_1, reinterpret_tensor(buf37, (128, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf38)
    del arg26_1
    del arg27_1
    buf39 = buf24; del buf24  # reuse
    buf40 = buf23; del buf23  # reuse
    buf42 = reinterpret_tensor(buf37, (1, 128, 768), (98304, 768, 1), 0); del buf37  # reuse
    cpp_fused_add_native_layer_norm_12(c_void_p(buf38.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()))
    del arg28_1
    del arg29_1
    buf43 = reinterpret_tensor(buf21, (128, 3072), (3072, 1), 0); del buf21  # reuse
    # Source Nodes: [x_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg31_1, reinterpret_tensor(buf42, (128, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf43)
    del arg30_1
    del arg31_1
    buf44 = reinterpret_tensor(buf43, (1, 128, 3072), (393216, 3072, 1), 0); del buf43  # reuse
    cpp_fused_gelu_13(c_void_p(buf44.data_ptr()))
    buf45 = buf38; del buf38  # reuse
    # Source Nodes: [x_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg33_1, reinterpret_tensor(buf44, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg32_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf45)
    del arg32_1
    del arg33_1
    buf46 = buf40; del buf40  # reuse
    buf47 = buf39; del buf39  # reuse
    buf49 = buf26; del buf26  # reuse
    cpp_fused_add_native_layer_norm_14(c_void_p(buf45.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()))
    del arg34_1
    del arg35_1
    buf50 = buf45; del buf45  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_2_attention_q_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg37_1, reinterpret_tensor(buf49, (128, 768), (768, 1), 0), reinterpret_tensor(arg36_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf50)
    del arg36_1
    del arg37_1
    buf51 = reinterpret_tensor(buf42, (128, 768), (768, 1), 0); del buf42  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_2_attention_k_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg39_1, reinterpret_tensor(buf49, (128, 768), (768, 1), 0), reinterpret_tensor(arg38_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf51)
    del arg38_1
    del arg39_1
    buf52 = reinterpret_tensor(buf27, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf27  # reuse
    cpp_fused_div_15(c_void_p(buf50.data_ptr()), c_void_p(buf52.data_ptr()))
    buf53 = reinterpret_tensor(buf35, (12, 128, 128), (16384, 128, 1), 0); del buf35  # reuse
    # Source Nodes: [scores_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf52, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf51, (12, 64, 128), (64, 1, 768), 0), out=buf53)
    buf54 = buf33; del buf33  # reuse
    buf55 = reinterpret_tensor(buf53, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf53  # reuse
    buf56 = buf31; del buf31  # reuse
    cpp_fused__softmax_lift_fresh_masked_fill_16(c_void_p(buf55.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()))
    buf57 = reinterpret_tensor(buf52, (128, 768), (768, 1), 0); del buf52  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_2_attention_v_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg41_1, reinterpret_tensor(buf49, (128, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf57)
    del arg40_1
    del arg41_1
    buf58 = buf55; del buf55  # reuse
    cpp_fused__softmax_17(c_void_p(buf58.data_ptr()), c_void_p(buf56.data_ptr()))
    buf59 = reinterpret_tensor(buf51, (12, 128, 64), (8192, 64, 1), 0); del buf51  # reuse
    # Source Nodes: [context_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf58, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf57, (12, 128, 64), (64, 768, 1), 0), out=buf59)
    buf60 = reinterpret_tensor(buf57, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf57  # reuse
    cpp_fused_clone_18(c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    buf61 = reinterpret_tensor(buf59, (128, 768), (768, 1), 0); del buf59  # reuse
    # Source Nodes: [sa_output_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg43_1, reinterpret_tensor(buf60, (128, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf61)
    del arg42_1
    del arg43_1
    buf62 = buf47; del buf47  # reuse
    buf63 = buf46; del buf46  # reuse
    buf65 = reinterpret_tensor(buf60, (1, 128, 768), (98304, 768, 1), 0); del buf60  # reuse
    cpp_fused_add_native_layer_norm_19(c_void_p(buf61.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf65.data_ptr()))
    del arg44_1
    del arg45_1
    buf66 = reinterpret_tensor(buf44, (128, 3072), (3072, 1), 0); del buf44  # reuse
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg47_1, reinterpret_tensor(buf65, (128, 768), (768, 1), 0), reinterpret_tensor(arg46_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf66)
    del arg46_1
    del arg47_1
    buf67 = reinterpret_tensor(buf66, (1, 128, 3072), (393216, 3072, 1), 0); del buf66  # reuse
    cpp_fused_gelu_20(c_void_p(buf67.data_ptr()))
    buf68 = buf61; del buf61  # reuse
    # Source Nodes: [x_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg49_1, reinterpret_tensor(buf67, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg48_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf68)
    del arg48_1
    del arg49_1
    buf69 = buf63; del buf63  # reuse
    buf70 = buf62; del buf62  # reuse
    buf72 = buf49; del buf49  # reuse
    cpp_fused_add_native_layer_norm_21(c_void_p(buf68.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()))
    del arg50_1
    del arg51_1
    buf73 = buf68; del buf68  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_3_attention_q_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg53_1, reinterpret_tensor(buf72, (128, 768), (768, 1), 0), reinterpret_tensor(arg52_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf73)
    del arg52_1
    del arg53_1
    buf74 = reinterpret_tensor(buf65, (128, 768), (768, 1), 0); del buf65  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_3_attention_k_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg55_1, reinterpret_tensor(buf72, (128, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf74)
    del arg54_1
    del arg55_1
    buf75 = reinterpret_tensor(buf50, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf50  # reuse
    cpp_fused_div_22(c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()))
    buf76 = reinterpret_tensor(buf58, (12, 128, 128), (16384, 128, 1), 0); del buf58  # reuse
    # Source Nodes: [scores_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf75, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf74, (12, 64, 128), (64, 1, 768), 0), out=buf76)
    buf77 = buf56; del buf56  # reuse
    buf78 = reinterpret_tensor(buf76, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf76  # reuse
    buf79 = buf54; del buf54  # reuse
    cpp_fused__softmax_lift_fresh_masked_fill_23(c_void_p(buf78.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf79.data_ptr()))
    buf80 = reinterpret_tensor(buf75, (128, 768), (768, 1), 0); del buf75  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_3_attention_v_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg57_1, reinterpret_tensor(buf72, (128, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf80)
    del arg56_1
    del arg57_1
    buf81 = buf78; del buf78  # reuse
    cpp_fused__softmax_24(c_void_p(buf81.data_ptr()), c_void_p(buf79.data_ptr()))
    buf82 = reinterpret_tensor(buf74, (12, 128, 64), (8192, 64, 1), 0); del buf74  # reuse
    # Source Nodes: [context_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf81, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf80, (12, 128, 64), (64, 768, 1), 0), out=buf82)
    buf83 = reinterpret_tensor(buf80, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf80  # reuse
    cpp_fused_clone_25(c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    buf84 = reinterpret_tensor(buf82, (128, 768), (768, 1), 0); del buf82  # reuse
    # Source Nodes: [sa_output_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg59_1, reinterpret_tensor(buf83, (128, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf84)
    del arg58_1
    del arg59_1
    buf85 = buf70; del buf70  # reuse
    buf86 = buf69; del buf69  # reuse
    buf88 = reinterpret_tensor(buf83, (1, 128, 768), (98304, 768, 1), 0); del buf83  # reuse
    cpp_fused_add_native_layer_norm_26(c_void_p(buf84.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()))
    del arg60_1
    del arg61_1
    buf89 = reinterpret_tensor(buf67, (128, 3072), (3072, 1), 0); del buf67  # reuse
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg63_1, reinterpret_tensor(buf88, (128, 768), (768, 1), 0), reinterpret_tensor(arg62_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf89)
    del arg62_1
    del arg63_1
    buf90 = reinterpret_tensor(buf89, (1, 128, 3072), (393216, 3072, 1), 0); del buf89  # reuse
    cpp_fused_gelu_27(c_void_p(buf90.data_ptr()))
    buf91 = buf84; del buf84  # reuse
    # Source Nodes: [x_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg65_1, reinterpret_tensor(buf90, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg64_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf91)
    del arg64_1
    del arg65_1
    buf92 = buf86; del buf86  # reuse
    buf93 = buf85; del buf85  # reuse
    buf95 = buf72; del buf72  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf91.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()))
    del arg66_1
    del arg67_1
    buf96 = buf91; del buf91  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_4_attention_q_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg69_1, reinterpret_tensor(buf95, (128, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf96)
    del arg68_1
    del arg69_1
    buf97 = reinterpret_tensor(buf88, (128, 768), (768, 1), 0); del buf88  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_4_attention_k_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg71_1, reinterpret_tensor(buf95, (128, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf97)
    del arg70_1
    del arg71_1
    buf98 = reinterpret_tensor(buf73, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf73  # reuse
    cpp_fused_div_29(c_void_p(buf96.data_ptr()), c_void_p(buf98.data_ptr()))
    buf99 = reinterpret_tensor(buf81, (12, 128, 128), (16384, 128, 1), 0); del buf81  # reuse
    # Source Nodes: [scores_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf98, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf97, (12, 64, 128), (64, 1, 768), 0), out=buf99)
    buf100 = buf79; del buf79  # reuse
    buf101 = reinterpret_tensor(buf99, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf99  # reuse
    buf102 = buf77; del buf77  # reuse
    cpp_fused__softmax_lift_fresh_masked_fill_30(c_void_p(buf101.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()))
    buf103 = reinterpret_tensor(buf98, (128, 768), (768, 1), 0); del buf98  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_4_attention_v_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg73_1, reinterpret_tensor(buf95, (128, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf103)
    del arg72_1
    del arg73_1
    buf104 = buf101; del buf101  # reuse
    cpp_fused__softmax_31(c_void_p(buf104.data_ptr()), c_void_p(buf102.data_ptr()))
    buf105 = reinterpret_tensor(buf97, (12, 128, 64), (8192, 64, 1), 0); del buf97  # reuse
    # Source Nodes: [context_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf104, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf103, (12, 128, 64), (64, 768, 1), 0), out=buf105)
    buf106 = reinterpret_tensor(buf103, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf103  # reuse
    cpp_fused_clone_32(c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    buf107 = reinterpret_tensor(buf105, (128, 768), (768, 1), 0); del buf105  # reuse
    # Source Nodes: [sa_output_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg75_1, reinterpret_tensor(buf106, (128, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf107)
    del arg74_1
    del arg75_1
    buf108 = buf93; del buf93  # reuse
    buf109 = buf92; del buf92  # reuse
    buf111 = reinterpret_tensor(buf106, (1, 128, 768), (98304, 768, 1), 0); del buf106  # reuse
    cpp_fused_add_native_layer_norm_33(c_void_p(buf107.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()))
    del arg76_1
    del arg77_1
    buf112 = reinterpret_tensor(buf90, (128, 3072), (3072, 1), 0); del buf90  # reuse
    # Source Nodes: [x_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg79_1, reinterpret_tensor(buf111, (128, 768), (768, 1), 0), reinterpret_tensor(arg78_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf112)
    del arg78_1
    del arg79_1
    buf113 = reinterpret_tensor(buf112, (1, 128, 3072), (393216, 3072, 1), 0); del buf112  # reuse
    cpp_fused_gelu_34(c_void_p(buf113.data_ptr()))
    buf114 = reinterpret_tensor(buf95, (128, 768), (768, 1), 0); del buf95  # reuse
    # Source Nodes: [x_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg81_1, reinterpret_tensor(buf113, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg80_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf114)
    del arg80_1
    del arg81_1
    buf115 = buf109; del buf109  # reuse
    buf116 = buf108; del buf108  # reuse
    buf118 = reinterpret_tensor(buf107, (1, 128, 768), (98304, 768, 1), 0); del buf107  # reuse
    cpp_fused_add_native_layer_norm_35(c_void_p(buf114.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf118.data_ptr()))
    del arg82_1
    del arg83_1
    buf119 = buf114; del buf114  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_5_attention_q_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf118, (128, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf119)
    del arg84_1
    del arg85_1
    buf120 = reinterpret_tensor(buf111, (128, 768), (768, 1), 0); del buf111  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_5_attention_k_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg87_1, reinterpret_tensor(buf118, (128, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf120)
    del arg86_1
    del arg87_1
    buf121 = reinterpret_tensor(buf96, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf96  # reuse
    cpp_fused_div_36(c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()))
    del buf119
    buf122 = reinterpret_tensor(buf104, (12, 128, 128), (16384, 128, 1), 0); del buf104  # reuse
    # Source Nodes: [scores_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf121, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf120, (12, 64, 128), (64, 1, 768), 0), out=buf122)
    buf123 = buf102; del buf102  # reuse
    buf124 = reinterpret_tensor(buf122, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf122  # reuse
    buf125 = buf100; del buf100  # reuse
    cpp_fused__softmax_lift_fresh_masked_fill_37(c_void_p(buf124.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf125.data_ptr()))
    del buf123
    buf126 = reinterpret_tensor(buf121, (128, 768), (768, 1), 0); del buf121  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_5_attention_v_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg89_1, reinterpret_tensor(buf118, (128, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf126)
    del arg88_1
    del arg89_1
    buf127 = buf124; del buf124  # reuse
    cpp_fused__softmax_38(c_void_p(buf127.data_ptr()), c_void_p(buf125.data_ptr()))
    del buf125
    buf128 = reinterpret_tensor(buf120, (12, 128, 64), (8192, 64, 1), 0); del buf120  # reuse
    # Source Nodes: [context_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf127, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf126, (12, 128, 64), (64, 768, 1), 0), out=buf128)
    del buf127
    buf129 = reinterpret_tensor(buf126, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf126  # reuse
    cpp_fused_clone_39(c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    buf130 = reinterpret_tensor(buf128, (128, 768), (768, 1), 0); del buf128  # reuse
    # Source Nodes: [sa_output_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf129, (128, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf130)
    del arg90_1
    del arg91_1
    buf131 = buf116; del buf116  # reuse
    buf132 = buf115; del buf115  # reuse
    buf134 = reinterpret_tensor(buf129, (1, 128, 768), (98304, 768, 1), 0); del buf129  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf130.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()))
    del arg92_1
    del arg93_1
    buf135 = reinterpret_tensor(buf113, (128, 3072), (3072, 1), 0); del buf113  # reuse
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg95_1, reinterpret_tensor(buf134, (128, 768), (768, 1), 0), reinterpret_tensor(arg94_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf135)
    del arg94_1
    del arg95_1
    buf136 = reinterpret_tensor(buf135, (1, 128, 3072), (393216, 3072, 1), 0); del buf135  # reuse
    cpp_fused_gelu_41(c_void_p(buf136.data_ptr()))
    buf137 = buf130; del buf130  # reuse
    # Source Nodes: [x_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg97_1, reinterpret_tensor(buf136, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg96_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf137)
    del arg96_1
    del arg97_1
    del buf136
    buf138 = buf132; del buf132  # reuse
    buf139 = buf131; del buf131  # reuse
    buf141 = buf118; del buf118  # reuse
    cpp_fused_add_native_layer_norm_42(c_void_p(buf137.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()))
    del arg98_1
    del arg99_1
    del buf134
    buf142 = buf137; del buf137  # reuse
    # Source Nodes: [prediction_logits], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg101_1, reinterpret_tensor(buf141, (128, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf142)
    del arg100_1
    del arg101_1
    buf143 = buf139; del buf139  # reuse
    buf144 = buf138; del buf138  # reuse
    buf146 = buf141; del buf141  # reuse
    cpp_fused_gelu_native_layer_norm_43(c_void_p(buf142.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()))
    del arg102_1
    del arg103_1
    del buf142
    buf147 = empty((128, 30522), device='cpu', dtype=torch.float32)
    # Source Nodes: [prediction_logits_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg105_1, reinterpret_tensor(buf146, (128, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf147)
    del arg104_1
    del arg105_1
    del buf146
    buf148 = reinterpret_tensor(buf144, (128, 1), (1, 128), 0); del buf144  # reuse
    buf149 = reinterpret_tensor(buf143, (128, 1), (1, 128), 0); del buf143  # reuse
    buf150 = empty((), device='cpu', dtype=torch.float32)
    buf151 = empty((), device='cpu', dtype=torch.int64)
    buf152 = buf150; del buf150  # reuse
    cpp_fused__log_softmax_nll_loss_forward_44(c_void_p(buf152.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()))
    del arg108_1
    return (buf152, reinterpret_tensor(buf147, (1, 128, 30522), (3906816, 30522, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((30522, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg107_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    arg108_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistilBertForMaskedLM', benchmark_compiled_module)
