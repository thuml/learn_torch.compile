
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp25.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_div_transpose_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_eq_lift_fresh_masked_fill_ones_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = static_cast<float>(1.0);
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = tmp0 == tmp1;
            out_ptr0[static_cast<long>(x0)] = tmp2;
        }
    }
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp5 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp11 = static_cast<float>(768.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-12);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                auto tmp21 = tmp19 + tmp20;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_div_transpose_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_masked_fill_8 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp11 = static_cast<float>(768.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-12);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                auto tmp21 = tmp19 + tmp20;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp11 = static_cast<float>(768.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-12);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                auto tmp21 = tmp19 + tmp20;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_div_transpose_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_masked_fill_14 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp11 = static_cast<float>(768.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-12);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                auto tmp21 = tmp19 + tmp20;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp11 = static_cast<float>(768.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-12);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                auto tmp21 = tmp19 + tmp20;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_div_transpose_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_masked_fill_20 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp11 = static_cast<float>(768.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-12);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                auto tmp21 = tmp19 + tmp20;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp11 = static_cast<float>(768.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-12);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                auto tmp21 = tmp19 + tmp20;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_div_transpose_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_masked_fill_26 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp11 = static_cast<float>(768.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-12);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                auto tmp21 = tmp19 + tmp20;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp11 = static_cast<float>(768.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-12);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                auto tmp21 = tmp19 + tmp20;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_div_transpose_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_masked_fill_32 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp11 = static_cast<float>(768.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-12);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                auto tmp21 = tmp19 + tmp20;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp11 = static_cast<float>(768.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-12);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                auto tmp21 = tmp19 + tmp20;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused__log_softmax_add_clamp_clone_div_embedding_native_layer_norm_native_layer_norm_backward_nll_loss_backward_nll_loss_forward_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
                       float* in_out_ptr4,
                       float* in_out_ptr5,
                       float* in_out_ptr6,
                       float* in_out_ptr7,
                       float* in_out_ptr8,
                       float* in_out_ptr9,
                       float* in_out_ptr10,
                       float* in_out_ptr11,
                       float* in_out_ptr12,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const long* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       bool* out_ptr8,
                       bool* out_ptr9,
                       float* out_ptr10,
                       bool* out_ptr11,
                       long* out_ptr12,
                       bool* out_ptr13,
                       long* out_ptr14)
{
    {
        {
            #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
            #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(1L + (2L*x0) + (2L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
            out_ptr3[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = out_ptr1[static_cast<long>(0L)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp4 = tmp3.exp();
                tmp_acc0_vec = tmp_acc0_vec + tmp4;
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr4[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = out_ptr1[static_cast<long>(0L)];
            auto tmp4 = out_ptr4[static_cast<long>(0L)];
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 - tmp2;
            auto tmp5 = std::log(tmp4);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp3 - tmp6;
            tmp7.store(out_ptr5 + static_cast<long>(x0));
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                auto tmp1 = out_ptr3[static_cast<long>(0L)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp4 = tmp3.exp();
                tmp_acc0_vec = tmp_acc0_vec + tmp4;
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr6[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
            auto tmp1 = out_ptr3[static_cast<long>(0L)];
            auto tmp4 = out_ptr6[static_cast<long>(0L)];
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 - tmp2;
            auto tmp5 = std::log(tmp4);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp3 - tmp6;
            tmp7.store(out_ptr7 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr1[static_cast<long>(0L)];
        auto tmp6 = in_ptr2[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(0);
        auto tmp2 = max_propagate_nan(tmp0, tmp1);
        auto tmp3 = static_cast<long>(128);
        auto tmp4 = min_propagate_nan(tmp2, tmp3);
        auto tmp5 = tmp4 != tmp3;
        auto tmp7 = max_propagate_nan(tmp6, tmp1);
        auto tmp8 = min_propagate_nan(tmp7, tmp3);
        auto tmp9 = tmp8 != tmp3;
        auto tmp10 = tmp5 ? tmp4 : tmp1;
        auto tmp11 = decltype(tmp10)(tmp10 + 128);
        auto tmp12 = tmp10 < 0;
        auto tmp13 = tmp12 ? tmp11 : tmp10;
        TORCH_CHECK((0 <= tmp13) & (tmp13 < 128L), "index out of bounds: 0 <= tmp13 < 128L")
        auto tmp14 = out_ptr5[static_cast<long>(tmp13)];
        auto tmp15 = decltype(tmp14)(-tmp14);
        auto tmp16 = static_cast<float>(0.0);
        auto tmp17 = tmp5 ? tmp15 : tmp16;
        auto tmp18 = c10::convert<long>(tmp5);
        auto tmp19 = c10::convert<float>(tmp18);
        auto tmp20 = tmp17 / tmp19;
        auto tmp21 = tmp9 ? tmp8 : tmp1;
        auto tmp22 = decltype(tmp21)(tmp21 + 128);
        auto tmp23 = tmp21 < 0;
        auto tmp24 = tmp23 ? tmp22 : tmp21;
        TORCH_CHECK((0 <= tmp24) & (tmp24 < 128L), "index out of bounds: 0 <= tmp24 < 128L")
        auto tmp25 = out_ptr7[static_cast<long>(tmp24)];
        auto tmp26 = decltype(tmp25)(-tmp25);
        auto tmp27 = tmp9 ? tmp26 : tmp16;
        auto tmp28 = c10::convert<long>(tmp9);
        auto tmp29 = c10::convert<float>(tmp28);
        auto tmp30 = tmp27 / tmp29;
        auto tmp31 = decltype(tmp20)(tmp20 + tmp30);
        auto tmp32 = static_cast<float>(2.0);
        auto tmp33 = tmp31 / tmp32;
        out_ptr8[static_cast<long>(0L)] = tmp5;
        out_ptr9[static_cast<long>(0L)] = tmp9;
        out_ptr10[static_cast<long>(0L)] = tmp33;
        out_ptr11[static_cast<long>(0L)] = tmp9;
        out_ptr12[static_cast<long>(0L)] = tmp21;
        out_ptr13[static_cast<long>(0L)] = tmp5;
        out_ptr14[static_cast<long>(0L)] = tmp10;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr3 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr4 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr5 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr6 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr7 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr8 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr9 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr10 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr11 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-12);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr12 + static_cast<long>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106 = args
    args.clear()
    assert_size_stride(primals_1, (30522, 768), (768, 1))
    assert_size_stride(primals_2, (512, 768), (768, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, 768), (768, 1))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, 768), (768, 1))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, 768), (768, 1))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (768, 768), (768, 1))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (3072, 768), (768, 1))
    assert_size_stride(primals_16, (3072, ), (1, ))
    assert_size_stride(primals_17, (768, 3072), (3072, 1))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, 768), (768, 1))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, 768), (768, 1))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, 768), (768, 1))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (768, 768), (768, 1))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (3072, 768), (768, 1))
    assert_size_stride(primals_32, (3072, ), (1, ))
    assert_size_stride(primals_33, (768, 3072), (3072, 1))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, 768), (768, 1))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, 768), (768, 1))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, 768), (768, 1))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, 768), (768, 1))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (3072, 768), (768, 1))
    assert_size_stride(primals_48, (3072, ), (1, ))
    assert_size_stride(primals_49, (768, 3072), (3072, 1))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, 768), (768, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, 768), (768, 1))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, 768), (768, 1))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (768, 768), (768, 1))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (3072, 768), (768, 1))
    assert_size_stride(primals_64, (3072, ), (1, ))
    assert_size_stride(primals_65, (768, 3072), (3072, 1))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, 768), (768, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, 768), (768, 1))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, 768), (768, 1))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (768, 768), (768, 1))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (3072, 768), (768, 1))
    assert_size_stride(primals_80, (3072, ), (1, ))
    assert_size_stride(primals_81, (768, 3072), (3072, 1))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, 768), (768, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, 768), (768, 1))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, 768), (768, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (768, 768), (768, 1))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (3072, 768), (768, 1))
    assert_size_stride(primals_96, (3072, ), (1, ))
    assert_size_stride(primals_97, (768, 3072), (3072, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (2, 768), (768, 1))
    assert_size_stride(primals_102, (2, ), (1, ))
    assert_size_stride(primals_103, (1, 512), (512, 1))
    assert_size_stride(primals_104, (1, 128), (128, 1))
    assert_size_stride(primals_105, (1, ), (1, ))
    assert_size_stride(primals_106, (1, ), (1, ))
    buf0 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    buf4 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_0(c_void_p(primals_104.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()))
    del primals_1
    del primals_2
    del primals_4
    # Source Nodes: [embeddings_1, hidden_state], Original ATen: [aten.native_dropout, aten.native_layer_norm]
    buf5 = aten.native_dropout(buf4, 0.1, True)
    buf6 = buf5[0]
    buf7 = buf5[1]
    del buf5
    buf8 = reinterpret_tensor(buf4, (128, 768), (768, 1), 0); del buf4  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_0_attention_q_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_6, reinterpret_tensor(buf6, (128, 768), (768, 1), 0), reinterpret_tensor(primals_5, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf8)
    del primals_6
    buf9 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_0_attention_k_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_8, reinterpret_tensor(buf6, (128, 768), (768, 1), 0), reinterpret_tensor(primals_7, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf9)
    del primals_8
    buf10 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_0_attention_v_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_10, reinterpret_tensor(buf6, (128, 768), (768, 1), 0), reinterpret_tensor(primals_9, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf10)
    del primals_10
    buf11 = empty((1, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf230 = empty_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_div_transpose_1(c_void_p(buf8.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf230.data_ptr()))
    buf12 = empty((12, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf11, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf9, (12, 64, 128), (64, 1, 768), 0), out=buf12)
    buf13 = empty((1, 1, 1, 128), device='cpu', dtype=torch.bool)
    buf14 = empty_strided((1, 12, 128, 1), (1536, 128, 1, 1536), device='cpu', dtype=torch.float32)
    buf15 = reinterpret_tensor(buf12, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf12  # reuse
    buf16 = empty_strided((1, 12, 128, 1), (1536, 128, 1, 1536), device='cpu', dtype=torch.float32)
    buf17 = buf15; del buf15  # reuse
    cpp_fused__softmax_eq_lift_fresh_masked_fill_ones_view_2(c_void_p(buf17.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf16.data_ptr()))
    # Source Nodes: [weights, weights_1], Original ATen: [aten._softmax, aten.native_dropout]
    buf18 = aten.native_dropout(buf17, 0.1, True)
    buf19 = buf18[0]
    buf20 = buf18[1]
    del buf18
    buf21 = reinterpret_tensor(buf11, (12, 128, 64), (8192, 64, 1), 0); del buf11  # reuse
    # Source Nodes: [context], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf19, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf10, (12, 128, 64), (64, 768, 1), 0), out=buf21)
    buf22 = buf8; del buf8  # reuse
    cpp_fused_view_3(c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    buf23 = reinterpret_tensor(buf21, (128, 768), (768, 1), 0); del buf21  # reuse
    # Source Nodes: [sa_output], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_12, buf22, reinterpret_tensor(primals_11, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf23)
    del primals_12
    buf24 = buf0; del buf0  # reuse
    buf25 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf27 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    buf28 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_4(c_void_p(buf23.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    buf29 = empty((128, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_16, buf28, reinterpret_tensor(primals_15, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf29)
    del primals_16
    buf30 = empty((128, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_5(c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()))
    buf31 = buf23; del buf23  # reuse
    # Source Nodes: [x_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_18, buf30, reinterpret_tensor(primals_17, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf31)
    del primals_18
    # Source Nodes: [ffn_output], Original ATen: [aten.native_dropout]
    buf32 = aten.native_dropout(reinterpret_tensor(buf31, (1, 128, 768), (98304, 768, 1), 0), 0.1, True)
    buf33 = buf32[0]
    buf34 = buf32[1]
    del buf32
    buf35 = buf24; del buf24  # reuse
    buf36 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf38 = reinterpret_tensor(buf31, (1, 128, 768), (98304, 768, 1), 0); del buf31  # reuse
    buf39 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_6(c_void_p(buf33.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    del primals_14
    buf40 = reinterpret_tensor(buf33, (128, 768), (768, 1), 0); del buf33  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_1_attention_q_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_22, buf39, reinterpret_tensor(primals_21, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf40)
    del primals_22
    buf41 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_1_attention_k_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_24, buf39, reinterpret_tensor(primals_23, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf41)
    del primals_24
    buf42 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_1_attention_v_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_26, buf39, reinterpret_tensor(primals_25, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf42)
    del primals_26
    buf43 = empty((1, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf227 = empty_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_div_transpose_7(c_void_p(buf40.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf227.data_ptr()))
    buf44 = empty((12, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf43, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf41, (12, 64, 128), (64, 1, 768), 0), out=buf44)
    buf45 = buf16; del buf16  # reuse
    buf46 = reinterpret_tensor(buf44, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf44  # reuse
    buf47 = buf14; del buf14  # reuse
    buf48 = buf46; del buf46  # reuse
    cpp_fused__softmax_lift_fresh_masked_fill_8(c_void_p(buf48.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()))
    # Source Nodes: [weights_2, weights_3], Original ATen: [aten._softmax, aten.native_dropout]
    buf49 = aten.native_dropout(buf48, 0.1, True)
    buf50 = buf49[0]
    buf51 = buf49[1]
    del buf49
    buf52 = reinterpret_tensor(buf43, (12, 128, 64), (8192, 64, 1), 0); del buf43  # reuse
    # Source Nodes: [context_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf50, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf42, (12, 128, 64), (64, 768, 1), 0), out=buf52)
    buf53 = buf40; del buf40  # reuse
    cpp_fused_view_9(c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()))
    buf54 = reinterpret_tensor(buf52, (128, 768), (768, 1), 0); del buf52  # reuse
    # Source Nodes: [sa_output_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_28, buf53, reinterpret_tensor(primals_27, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf54)
    del primals_28
    buf55 = buf35; del buf35  # reuse
    buf56 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf58 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    buf59 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_10(c_void_p(buf54.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()))
    del primals_20
    buf60 = empty((128, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_32, buf59, reinterpret_tensor(primals_31, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf60)
    del primals_32
    buf61 = empty((128, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_11(c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    buf62 = buf54; del buf54  # reuse
    # Source Nodes: [x_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_34, buf61, reinterpret_tensor(primals_33, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf62)
    del primals_34
    # Source Nodes: [ffn_output_2], Original ATen: [aten.native_dropout]
    buf63 = aten.native_dropout(reinterpret_tensor(buf62, (1, 128, 768), (98304, 768, 1), 0), 0.1, True)
    buf64 = buf63[0]
    buf65 = buf63[1]
    del buf63
    buf66 = buf55; del buf55  # reuse
    buf67 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf69 = reinterpret_tensor(buf62, (1, 128, 768), (98304, 768, 1), 0); del buf62  # reuse
    buf70 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_12(c_void_p(buf64.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del primals_30
    buf71 = reinterpret_tensor(buf64, (128, 768), (768, 1), 0); del buf64  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_2_attention_q_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_38, buf70, reinterpret_tensor(primals_37, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf71)
    del primals_38
    buf72 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_2_attention_k_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_40, buf70, reinterpret_tensor(primals_39, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf72)
    del primals_40
    buf73 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_2_attention_v_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_42, buf70, reinterpret_tensor(primals_41, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf73)
    del primals_42
    buf74 = empty((1, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf224 = empty_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_div_transpose_13(c_void_p(buf71.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf224.data_ptr()))
    buf75 = empty((12, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf74, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf72, (12, 64, 128), (64, 1, 768), 0), out=buf75)
    buf76 = buf47; del buf47  # reuse
    buf77 = reinterpret_tensor(buf75, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf75  # reuse
    buf78 = buf45; del buf45  # reuse
    buf79 = buf77; del buf77  # reuse
    cpp_fused__softmax_lift_fresh_masked_fill_14(c_void_p(buf79.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()))
    # Source Nodes: [weights_4, weights_5], Original ATen: [aten._softmax, aten.native_dropout]
    buf80 = aten.native_dropout(buf79, 0.1, True)
    buf81 = buf80[0]
    buf82 = buf80[1]
    del buf80
    buf83 = reinterpret_tensor(buf74, (12, 128, 64), (8192, 64, 1), 0); del buf74  # reuse
    # Source Nodes: [context_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf81, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf73, (12, 128, 64), (64, 768, 1), 0), out=buf83)
    buf84 = buf71; del buf71  # reuse
    cpp_fused_view_15(c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    buf85 = reinterpret_tensor(buf83, (128, 768), (768, 1), 0); del buf83  # reuse
    # Source Nodes: [sa_output_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_44, buf84, reinterpret_tensor(primals_43, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf85)
    del primals_44
    buf86 = buf66; del buf66  # reuse
    buf87 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf89 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    buf90 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_16(c_void_p(buf85.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del primals_36
    buf91 = empty((128, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_48, buf90, reinterpret_tensor(primals_47, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf91)
    del primals_48
    buf92 = empty((128, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_17(c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    buf93 = buf85; del buf85  # reuse
    # Source Nodes: [x_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_50, buf92, reinterpret_tensor(primals_49, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf93)
    del primals_50
    # Source Nodes: [ffn_output_4], Original ATen: [aten.native_dropout]
    buf94 = aten.native_dropout(reinterpret_tensor(buf93, (1, 128, 768), (98304, 768, 1), 0), 0.1, True)
    buf95 = buf94[0]
    buf96 = buf94[1]
    del buf94
    buf97 = buf86; del buf86  # reuse
    buf98 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf100 = reinterpret_tensor(buf93, (1, 128, 768), (98304, 768, 1), 0); del buf93  # reuse
    buf101 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_18(c_void_p(buf95.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    del primals_46
    buf102 = reinterpret_tensor(buf95, (128, 768), (768, 1), 0); del buf95  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_3_attention_q_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_54, buf101, reinterpret_tensor(primals_53, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf102)
    del primals_54
    buf103 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_3_attention_k_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_56, buf101, reinterpret_tensor(primals_55, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf103)
    del primals_56
    buf104 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_3_attention_v_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_58, buf101, reinterpret_tensor(primals_57, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf104)
    del primals_58
    buf105 = empty((1, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf221 = empty_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_div_transpose_19(c_void_p(buf102.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf221.data_ptr()))
    buf106 = empty((12, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf105, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf103, (12, 64, 128), (64, 1, 768), 0), out=buf106)
    buf107 = buf78; del buf78  # reuse
    buf108 = reinterpret_tensor(buf106, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf106  # reuse
    buf109 = buf76; del buf76  # reuse
    buf110 = buf108; del buf108  # reuse
    cpp_fused__softmax_lift_fresh_masked_fill_20(c_void_p(buf110.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()))
    # Source Nodes: [weights_6, weights_7], Original ATen: [aten._softmax, aten.native_dropout]
    buf111 = aten.native_dropout(buf110, 0.1, True)
    buf112 = buf111[0]
    buf113 = buf111[1]
    del buf111
    buf114 = reinterpret_tensor(buf105, (12, 128, 64), (8192, 64, 1), 0); del buf105  # reuse
    # Source Nodes: [context_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf112, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf104, (12, 128, 64), (64, 768, 1), 0), out=buf114)
    buf115 = buf102; del buf102  # reuse
    cpp_fused_view_21(c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    buf116 = reinterpret_tensor(buf114, (128, 768), (768, 1), 0); del buf114  # reuse
    # Source Nodes: [sa_output_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_60, buf115, reinterpret_tensor(primals_59, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf116)
    del primals_60
    buf117 = buf97; del buf97  # reuse
    buf118 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf120 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    buf121 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_22(c_void_p(buf116.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()))
    del primals_52
    buf122 = empty((128, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_64, buf121, reinterpret_tensor(primals_63, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf122)
    del primals_64
    buf123 = empty((128, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_23(c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    buf124 = buf116; del buf116  # reuse
    # Source Nodes: [x_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_66, buf123, reinterpret_tensor(primals_65, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf124)
    del primals_66
    # Source Nodes: [ffn_output_6], Original ATen: [aten.native_dropout]
    buf125 = aten.native_dropout(reinterpret_tensor(buf124, (1, 128, 768), (98304, 768, 1), 0), 0.1, True)
    buf126 = buf125[0]
    buf127 = buf125[1]
    del buf125
    buf128 = buf117; del buf117  # reuse
    buf129 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf131 = reinterpret_tensor(buf124, (1, 128, 768), (98304, 768, 1), 0); del buf124  # reuse
    buf132 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_24(c_void_p(buf126.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del primals_62
    buf133 = reinterpret_tensor(buf126, (128, 768), (768, 1), 0); del buf126  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_4_attention_q_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_70, buf132, reinterpret_tensor(primals_69, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf133)
    del primals_70
    buf134 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_4_attention_k_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_72, buf132, reinterpret_tensor(primals_71, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf134)
    del primals_72
    buf135 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_4_attention_v_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_74, buf132, reinterpret_tensor(primals_73, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf135)
    del primals_74
    buf136 = empty((1, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf218 = empty_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_div_transpose_25(c_void_p(buf133.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf218.data_ptr()))
    buf137 = empty((12, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf136, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf134, (12, 64, 128), (64, 1, 768), 0), out=buf137)
    buf138 = buf109; del buf109  # reuse
    buf139 = reinterpret_tensor(buf137, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf137  # reuse
    buf140 = buf107; del buf107  # reuse
    buf141 = buf139; del buf139  # reuse
    cpp_fused__softmax_lift_fresh_masked_fill_26(c_void_p(buf141.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf140.data_ptr()))
    # Source Nodes: [weights_8, weights_9], Original ATen: [aten._softmax, aten.native_dropout]
    buf142 = aten.native_dropout(buf141, 0.1, True)
    buf143 = buf142[0]
    buf144 = buf142[1]
    del buf142
    buf145 = reinterpret_tensor(buf136, (12, 128, 64), (8192, 64, 1), 0); del buf136  # reuse
    # Source Nodes: [context_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf143, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf135, (12, 128, 64), (64, 768, 1), 0), out=buf145)
    buf146 = buf133; del buf133  # reuse
    cpp_fused_view_27(c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    buf147 = reinterpret_tensor(buf145, (128, 768), (768, 1), 0); del buf145  # reuse
    # Source Nodes: [sa_output_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_76, buf146, reinterpret_tensor(primals_75, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf147)
    del primals_76
    buf148 = buf128; del buf128  # reuse
    buf149 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf151 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    buf152 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_28(c_void_p(buf147.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()))
    del primals_68
    buf153 = empty((128, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_80, buf152, reinterpret_tensor(primals_79, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf153)
    del primals_80
    buf154 = empty((128, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_29(c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    buf155 = buf147; del buf147  # reuse
    # Source Nodes: [x_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_82, buf154, reinterpret_tensor(primals_81, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf155)
    del primals_82
    # Source Nodes: [ffn_output_8], Original ATen: [aten.native_dropout]
    buf156 = aten.native_dropout(reinterpret_tensor(buf155, (1, 128, 768), (98304, 768, 1), 0), 0.1, True)
    buf157 = buf156[0]
    buf158 = buf156[1]
    del buf156
    buf159 = buf148; del buf148  # reuse
    buf160 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf162 = reinterpret_tensor(buf155, (1, 128, 768), (98304, 768, 1), 0); del buf155  # reuse
    buf163 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_30(c_void_p(buf157.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()))
    del primals_78
    buf164 = reinterpret_tensor(buf157, (128, 768), (768, 1), 0); del buf157  # reuse
    # Source Nodes: [l__mod___distilbert_transformer_layer_5_attention_q_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_86, buf163, reinterpret_tensor(primals_85, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf164)
    del primals_86
    buf165 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_5_attention_k_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_88, buf163, reinterpret_tensor(primals_87, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf165)
    del primals_88
    buf166 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___distilbert_transformer_layer_5_attention_v_lin], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_90, buf163, reinterpret_tensor(primals_89, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf166)
    del primals_90
    buf167 = empty((1, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf215 = empty_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_div_transpose_31(c_void_p(buf164.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf215.data_ptr()))
    buf168 = empty((12, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf167, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf165, (12, 64, 128), (64, 1, 768), 0), out=buf168)
    buf169 = buf140; del buf140  # reuse
    buf170 = reinterpret_tensor(buf168, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf168  # reuse
    buf171 = buf138; del buf138  # reuse
    buf172 = buf170; del buf170  # reuse
    cpp_fused__softmax_lift_fresh_masked_fill_32(c_void_p(buf172.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()))
    del buf169
    del buf171
    # Source Nodes: [weights_10, weights_11], Original ATen: [aten._softmax, aten.native_dropout]
    buf173 = aten.native_dropout(buf172, 0.1, True)
    buf174 = buf173[0]
    buf175 = buf173[1]
    del buf173
    buf176 = reinterpret_tensor(buf167, (12, 128, 64), (8192, 64, 1), 0); del buf167  # reuse
    # Source Nodes: [context_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf174, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf166, (12, 128, 64), (64, 768, 1), 0), out=buf176)
    buf177 = buf164; del buf164  # reuse
    cpp_fused_view_33(c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    buf178 = reinterpret_tensor(buf176, (128, 768), (768, 1), 0); del buf176  # reuse
    # Source Nodes: [sa_output_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_92, buf177, reinterpret_tensor(primals_91, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf178)
    del primals_92
    buf179 = buf159; del buf159  # reuse
    buf180 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf182 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    buf183 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_34(c_void_p(buf178.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    del primals_84
    buf184 = empty((128, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_96, buf183, reinterpret_tensor(primals_95, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf184)
    del primals_96
    buf185 = empty((128, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_35(c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    buf186 = buf178; del buf178  # reuse
    # Source Nodes: [x_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_98, buf185, reinterpret_tensor(primals_97, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf186)
    del primals_98
    # Source Nodes: [ffn_output_10], Original ATen: [aten.native_dropout]
    buf187 = aten.native_dropout(reinterpret_tensor(buf186, (1, 128, 768), (98304, 768, 1), 0), 0.1, True)
    buf188 = buf187[0]
    buf189 = buf187[1]
    del buf187
    buf190 = buf179; del buf179  # reuse
    buf191 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf193 = reinterpret_tensor(buf186, (1, 128, 768), (98304, 768, 1), 0); del buf186  # reuse
    buf194 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_36(c_void_p(buf188.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()))
    del buf188
    del primals_100
    del primals_94
    # Source Nodes: [hidden_states, hidden_states_1], Original ATen: [aten.native_dropout, aten.native_layer_norm]
    buf195 = aten.native_dropout(buf194, 0.1, True)
    del buf194
    buf196 = buf195[0]
    buf197 = buf195[1]
    del buf195
    buf198 = empty((128, 2), device='cpu', dtype=torch.float32)
    # Source Nodes: [logits], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_102, reinterpret_tensor(buf196, (128, 768), (768, 1), 0), reinterpret_tensor(primals_101, (768, 2), (1, 768), 0), alpha=1, beta=1, out=buf198)
    del primals_102
    buf199 = reinterpret_tensor(buf190, (1, 128), (128, 1), 0); del buf190  # reuse
    buf201 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf200 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf205 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf202 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf203 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf206 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf207 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf204 = empty((1, ), device='cpu', dtype=torch.bool)
    buf208 = empty((1, ), device='cpu', dtype=torch.bool)
    buf232 = empty((), device='cpu', dtype=torch.float32)
    buf209 = empty((1, 1), device='cpu', dtype=torch.bool)
    buf210 = empty((1, 1), device='cpu', dtype=torch.int64)
    buf211 = empty((1, 1), device='cpu', dtype=torch.bool)
    buf212 = empty((1, 1), device='cpu', dtype=torch.int64)
    buf213 = reinterpret_tensor(buf191, (1, 128, 1), (128, 1, 1), 0); del buf191  # reuse
    buf214 = reinterpret_tensor(buf180, (1, 128, 1), (128, 1, 1), 0); del buf180  # reuse
    buf216 = reinterpret_tensor(buf160, (1, 128, 1), (128, 1, 1), 0); del buf160  # reuse
    buf217 = reinterpret_tensor(buf149, (1, 128, 1), (128, 1, 1), 0); del buf149  # reuse
    buf219 = reinterpret_tensor(buf129, (1, 128, 1), (128, 1, 1), 0); del buf129  # reuse
    buf220 = reinterpret_tensor(buf118, (1, 128, 1), (128, 1, 1), 0); del buf118  # reuse
    buf222 = reinterpret_tensor(buf98, (1, 128, 1), (128, 1, 1), 0); del buf98  # reuse
    buf223 = reinterpret_tensor(buf87, (1, 128, 1), (128, 1, 1), 0); del buf87  # reuse
    buf225 = reinterpret_tensor(buf67, (1, 128, 1), (128, 1, 1), 0); del buf67  # reuse
    buf226 = reinterpret_tensor(buf56, (1, 128, 1), (128, 1, 1), 0); del buf56  # reuse
    buf228 = reinterpret_tensor(buf36, (1, 128, 1), (128, 1, 1), 0); del buf36  # reuse
    buf229 = reinterpret_tensor(buf25, (1, 128, 1), (128, 1, 1), 0); del buf25  # reuse
    buf231 = reinterpret_tensor(buf1, (1, 128, 1), (128, 1, 1), 0); del buf1  # reuse
    cpp_fused__log_softmax_add_clamp_clone_div_embedding_native_layer_norm_native_layer_norm_backward_nll_loss_backward_nll_loss_forward_37(c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()))
    del buf198
    del buf201
    del buf202
    del buf205
    del buf206
    del primals_105
    del primals_106
    return (buf232, buf199, buf200, primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_104, reinterpret_tensor(primals_103, (1, 128), (512, 1), 0), buf3, buf7, reinterpret_tensor(buf6, (128, 768), (768, 1), 0), buf13, buf20, buf22, buf27, buf28, buf29, buf30, buf34, buf38, buf39, buf51, buf53, buf58, buf59, buf60, buf61, buf65, buf69, buf70, buf82, buf84, buf89, buf90, buf91, buf92, buf96, buf100, buf101, buf113, buf115, buf120, buf121, buf122, buf123, buf127, buf131, buf132, buf144, buf146, buf151, buf152, buf153, buf154, buf158, buf162, buf163, buf175, buf177, buf182, buf183, buf184, buf185, buf189, buf193, buf197, reinterpret_tensor(buf196, (128, 768), (768, 1), 0), buf203, buf204, buf207, buf208, buf209, buf210, buf211, buf212, reinterpret_tensor(primals_101, (2, 768), (768, 1), 0), buf213, reinterpret_tensor(primals_97, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_95, (3072, 768), (768, 1), 0), buf214, reinterpret_tensor(primals_91, (768, 768), (768, 1), 0), reinterpret_tensor(buf174, (12, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf166, (12, 64, 128), (64, 1, 768), 0), buf172, buf215, reinterpret_tensor(buf165, (12, 128, 64), (64, 768, 1), 0), reinterpret_tensor(primals_89, (768, 768), (768, 1), 0), reinterpret_tensor(primals_87, (768, 768), (768, 1), 0), reinterpret_tensor(primals_85, (768, 768), (768, 1), 0), buf216, reinterpret_tensor(primals_81, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_79, (3072, 768), (768, 1), 0), buf217, reinterpret_tensor(primals_75, (768, 768), (768, 1), 0), reinterpret_tensor(buf143, (12, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf135, (12, 64, 128), (64, 1, 768), 0), buf141, buf218, reinterpret_tensor(buf134, (12, 128, 64), (64, 768, 1), 0), reinterpret_tensor(primals_73, (768, 768), (768, 1), 0), reinterpret_tensor(primals_71, (768, 768), (768, 1), 0), reinterpret_tensor(primals_69, (768, 768), (768, 1), 0), buf219, reinterpret_tensor(primals_65, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_63, (3072, 768), (768, 1), 0), buf220, reinterpret_tensor(primals_59, (768, 768), (768, 1), 0), reinterpret_tensor(buf112, (12, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf104, (12, 64, 128), (64, 1, 768), 0), buf110, buf221, reinterpret_tensor(buf103, (12, 128, 64), (64, 768, 1), 0), reinterpret_tensor(primals_57, (768, 768), (768, 1), 0), reinterpret_tensor(primals_55, (768, 768), (768, 1), 0), reinterpret_tensor(primals_53, (768, 768), (768, 1), 0), buf222, reinterpret_tensor(primals_49, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_47, (3072, 768), (768, 1), 0), buf223, reinterpret_tensor(primals_43, (768, 768), (768, 1), 0), reinterpret_tensor(buf81, (12, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf73, (12, 64, 128), (64, 1, 768), 0), buf79, buf224, reinterpret_tensor(buf72, (12, 128, 64), (64, 768, 1), 0), reinterpret_tensor(primals_41, (768, 768), (768, 1), 0), reinterpret_tensor(primals_39, (768, 768), (768, 1), 0), reinterpret_tensor(primals_37, (768, 768), (768, 1), 0), buf225, reinterpret_tensor(primals_33, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_31, (3072, 768), (768, 1), 0), buf226, reinterpret_tensor(primals_27, (768, 768), (768, 1), 0), reinterpret_tensor(buf50, (12, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf42, (12, 64, 128), (64, 1, 768), 0), buf48, buf227, reinterpret_tensor(buf41, (12, 128, 64), (64, 768, 1), 0), reinterpret_tensor(primals_25, (768, 768), (768, 1), 0), reinterpret_tensor(primals_23, (768, 768), (768, 1), 0), reinterpret_tensor(primals_21, (768, 768), (768, 1), 0), buf228, reinterpret_tensor(primals_17, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_15, (3072, 768), (768, 1), 0), buf229, reinterpret_tensor(primals_11, (768, 768), (768, 1), 0), reinterpret_tensor(buf19, (12, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf10, (12, 64, 128), (64, 1, 768), 0), buf17, buf230, reinterpret_tensor(buf9, (12, 128, 64), (64, 768, 1), 0), reinterpret_tensor(primals_9, (768, 768), (768, 1), 0), reinterpret_tensor(primals_7, (768, 768), (768, 1), 0), reinterpret_tensor(primals_5, (768, 768), (768, 1), 0), buf231, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_104 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    primals_105 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    primals_106 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistilBertForQuestionAnswering', benchmark_compiled_module)
