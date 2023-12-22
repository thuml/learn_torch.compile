
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


cpp_fused_embedding_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*tmp3)));
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_arange_cat_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(long* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<long>(x0);
            out_ptr0[static_cast<long>(x0)] = tmp0;
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(512);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = c10::convert<long>(x0);
                        auto tmp7 = c10::convert<double>(tmp6);
                        auto tmp8 = static_cast<double>(-1.0);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = static_cast<double>(512.0);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = c10::convert<float>(tmp11);
                        auto tmp13 = c10::convert<long>(x1);
                        auto tmp14 = c10::convert<double>(tmp13);
                        auto tmp15 = static_cast<double>(2.0);
                        auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                        auto tmp17 = static_cast<double>(0.0);
                        auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                        auto tmp19 = c10::convert<float>(tmp18);
                        auto tmp20 = static_cast<float>(1024.0);
                        auto tmp21 = tmp19 / tmp20;
                        auto tmp22 = static_cast<float>(10000.0);
                        auto tmp23 = std::pow(tmp22, tmp21);
                        auto tmp24 = 1 / tmp23;
                        auto tmp25 = static_cast<float>(1.0);
                        auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                        auto tmp27 = decltype(tmp12)(tmp12 * tmp26);
                        auto tmp28 = std::sin(tmp27);
                        return tmp28;
                    }
                    ;
                    auto tmp29 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp30 = tmp0 >= tmp3;
                    auto tmp31 = static_cast<long>(1024);
                    auto tmp32 = tmp0 < tmp31;
                    auto tmp33 = [&]
                    {
                        auto tmp34 = c10::convert<long>(x0);
                        auto tmp35 = c10::convert<double>(tmp34);
                        auto tmp36 = static_cast<double>(-1.0);
                        auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                        auto tmp38 = static_cast<double>(512.0);
                        auto tmp39 = decltype(tmp37)(tmp37 + tmp38);
                        auto tmp40 = c10::convert<float>(tmp39);
                        auto tmp41 = c10::convert<long>((-512L) + x1);
                        auto tmp42 = c10::convert<double>(tmp41);
                        auto tmp43 = static_cast<double>(2.0);
                        auto tmp44 = decltype(tmp42)(tmp42 * tmp43);
                        auto tmp45 = static_cast<double>(0.0);
                        auto tmp46 = decltype(tmp44)(tmp44 + tmp45);
                        auto tmp47 = c10::convert<float>(tmp46);
                        auto tmp48 = static_cast<float>(1024.0);
                        auto tmp49 = tmp47 / tmp48;
                        auto tmp50 = static_cast<float>(10000.0);
                        auto tmp51 = std::pow(tmp50, tmp49);
                        auto tmp52 = 1 / tmp51;
                        auto tmp53 = static_cast<float>(1.0);
                        auto tmp54 = decltype(tmp52)(tmp52 * tmp53);
                        auto tmp55 = decltype(tmp40)(tmp40 * tmp54);
                        auto tmp56 = std::cos(tmp55);
                        return tmp56;
                    }
                    ;
                    auto tmp57 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                    auto tmp58 = tmp4 ? tmp29 : tmp57;
                    out_ptr1[static_cast<long>(x1 + (1024L*x0))] = tmp58;
                }
            }
        }
    }
}
''')


cpp_fused_add_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_7 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_25 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_31 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_37 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_43 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_49 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_55 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_67 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_73 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_79 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_85 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_89 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_91 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_94 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_95 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_97 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_100 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_101 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_103 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_106 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_107 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_109 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_113 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_115 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_118 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_119 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_121 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_124 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_125 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_127 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_130 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_131 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_133 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_136 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_137 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp0 + tmp3;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_index_select_mul_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 + tmp4;
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(512L + x2 + x2_inner + (1023L*x1) + (524288L*x0) + (524288L*(c10::div_floor_integer((x2 + x2_inner + (1023L*x1)), 523776L))))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = static_cast<float>(0.125);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp12 = tmp11.exp();
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (32768L*x1)), static_cast<long>(32768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (64L*x1) + (1024L*x2)), static_cast<long>(1024L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0) + (16384L*x0_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_143 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_145 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_add_native_layer_norm_native_layer_norm_backward_nll_loss_forward_146 = async_compile.cpp('''
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
                       float* in_out_ptr13,
                       float* in_out_ptr14,
                       float* in_out_ptr15,
                       float* in_out_ptr16,
                       float* in_out_ptr17,
                       float* in_out_ptr18,
                       float* in_out_ptr19,
                       float* in_out_ptr20,
                       float* in_out_ptr21,
                       float* in_out_ptr22,
                       float* in_out_ptr23,
                       float* in_out_ptr24,
                       float* in_out_ptr25,
                       float* in_out_ptr26,
                       float* in_out_ptr27,
                       float* in_out_ptr28,
                       float* in_out_ptr29,
                       float* in_out_ptr30,
                       float* in_out_ptr31,
                       float* in_out_ptr32,
                       float* in_out_ptr33,
                       float* in_out_ptr34,
                       float* in_out_ptr35,
                       float* in_out_ptr36,
                       float* in_out_ptr37,
                       float* in_out_ptr38,
                       float* in_out_ptr39,
                       float* in_out_ptr40,
                       float* in_out_ptr41,
                       float* in_out_ptr42,
                       float* in_out_ptr43,
                       float* in_out_ptr44,
                       float* in_out_ptr45,
                       float* in_out_ptr46,
                       float* in_out_ptr47,
                       float* in_out_ptr48,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       long* out_ptr3,
                       float* out_ptr5)
{
    auto out_ptr4 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32000L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32000L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32000L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32000L*x0)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32000L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32000L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = std::log(tmp4);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp3 - tmp6;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (32000L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    long tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = c10::convert<long>(tmp2);
                        auto tmp4 = static_cast<long>(0);
                        auto tmp5 = tmp2 ? tmp0 : tmp4;
                        auto tmp6 = decltype(tmp5)(tmp5 + 32000);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 32000L), "index out of bounds: 0 <= tmp8 < 32000L")
                        auto tmp9 = out_ptr2[static_cast<long>(tmp8 + (32000L*x0))];
                        auto tmp10 = decltype(tmp9)(-tmp9);
                        auto tmp11 = static_cast<float>(0.0);
                        auto tmp12 = tmp2 ? tmp10 : tmp11;
                        tmp_acc0 = tmp_acc0 + tmp3;
                        tmp_acc1 = tmp_acc1 + tmp12;
                    }
                    out_ptr3[static_cast<long>(0L)] = tmp_acc0;
                    out_ptr4[static_cast<long>(0L)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = out_ptr3[static_cast<long>(0L)];
                auto tmp2 = out_ptr4[static_cast<long>(0L)];
                auto tmp1 = c10::convert<float>(tmp0);
                auto tmp3 = tmp2 / tmp1;
                out_ptr5[static_cast<long>(0L)] = tmp1;
                in_out_ptr0[static_cast<long>(0L)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr13 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr14 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr15 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr16 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr17 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr18 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr19 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr20 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr21 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr22 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr23 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr24 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr25 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr26 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr27 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr28 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr29 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr30 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr31 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr32 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr33 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr34 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr35 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr36 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr37 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr38 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr39 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr40 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr41 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr42 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr43 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr44 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr45 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr46 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr47 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr48 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365 = args
    args.clear()
    assert_size_stride(primals_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_2, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_3, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_4, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_5, (16, 64), (64, 1))
    assert_size_stride(primals_6, (16, 64), (64, 1))
    assert_size_stride(primals_7, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_8, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_9, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_10, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_11, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_12, (16, 64), (64, 1))
    assert_size_stride(primals_13, (16, 64), (64, 1))
    assert_size_stride(primals_14, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_15, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_16, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_17, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_18, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_19, (16, 64), (64, 1))
    assert_size_stride(primals_20, (16, 64), (64, 1))
    assert_size_stride(primals_21, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_22, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_23, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_24, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_25, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_26, (16, 64), (64, 1))
    assert_size_stride(primals_27, (16, 64), (64, 1))
    assert_size_stride(primals_28, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_29, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_30, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_31, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_32, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_33, (16, 64), (64, 1))
    assert_size_stride(primals_34, (16, 64), (64, 1))
    assert_size_stride(primals_35, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_36, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_37, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_38, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_39, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_40, (16, 64), (64, 1))
    assert_size_stride(primals_41, (16, 64), (64, 1))
    assert_size_stride(primals_42, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_43, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_44, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_45, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_46, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_47, (16, 64), (64, 1))
    assert_size_stride(primals_48, (16, 64), (64, 1))
    assert_size_stride(primals_49, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_50, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_51, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_52, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_53, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_54, (16, 64), (64, 1))
    assert_size_stride(primals_55, (16, 64), (64, 1))
    assert_size_stride(primals_56, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_57, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_58, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_59, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_60, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_61, (16, 64), (64, 1))
    assert_size_stride(primals_62, (16, 64), (64, 1))
    assert_size_stride(primals_63, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_64, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_65, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_66, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_67, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_68, (16, 64), (64, 1))
    assert_size_stride(primals_69, (16, 64), (64, 1))
    assert_size_stride(primals_70, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_71, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_72, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_73, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_74, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_75, (16, 64), (64, 1))
    assert_size_stride(primals_76, (16, 64), (64, 1))
    assert_size_stride(primals_77, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_78, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_79, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_80, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_81, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_82, (16, 64), (64, 1))
    assert_size_stride(primals_83, (16, 64), (64, 1))
    assert_size_stride(primals_84, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_85, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_86, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_87, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_88, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_89, (16, 64), (64, 1))
    assert_size_stride(primals_90, (16, 64), (64, 1))
    assert_size_stride(primals_91, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_92, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_93, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_94, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_95, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_96, (16, 64), (64, 1))
    assert_size_stride(primals_97, (16, 64), (64, 1))
    assert_size_stride(primals_98, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_99, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_100, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_101, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_102, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_103, (16, 64), (64, 1))
    assert_size_stride(primals_104, (16, 64), (64, 1))
    assert_size_stride(primals_105, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_106, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_107, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_108, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_109, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_110, (16, 64), (64, 1))
    assert_size_stride(primals_111, (16, 64), (64, 1))
    assert_size_stride(primals_112, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_113, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_114, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_115, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_116, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_117, (16, 64), (64, 1))
    assert_size_stride(primals_118, (16, 64), (64, 1))
    assert_size_stride(primals_119, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_120, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_121, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_122, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_123, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_124, (16, 64), (64, 1))
    assert_size_stride(primals_125, (16, 64), (64, 1))
    assert_size_stride(primals_126, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_127, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_128, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_129, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_130, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_131, (16, 64), (64, 1))
    assert_size_stride(primals_132, (16, 64), (64, 1))
    assert_size_stride(primals_133, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_134, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_135, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_136, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_137, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_138, (16, 64), (64, 1))
    assert_size_stride(primals_139, (16, 64), (64, 1))
    assert_size_stride(primals_140, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_141, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_142, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_143, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_144, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_145, (16, 64), (64, 1))
    assert_size_stride(primals_146, (16, 64), (64, 1))
    assert_size_stride(primals_147, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_148, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_149, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_150, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_151, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_152, (16, 64), (64, 1))
    assert_size_stride(primals_153, (16, 64), (64, 1))
    assert_size_stride(primals_154, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_155, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_156, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_157, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_158, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_159, (16, 64), (64, 1))
    assert_size_stride(primals_160, (16, 64), (64, 1))
    assert_size_stride(primals_161, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_162, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_163, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_164, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_165, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_166, (16, 64), (64, 1))
    assert_size_stride(primals_167, (16, 64), (64, 1))
    assert_size_stride(primals_168, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_169, (32000, 1024), (1024, 1))
    assert_size_stride(primals_170, (1024, ), (1, ))
    assert_size_stride(primals_171, (1024, ), (1, ))
    assert_size_stride(primals_172, (4096, 1024), (1024, 1))
    assert_size_stride(primals_173, (4096, ), (1, ))
    assert_size_stride(primals_174, (1024, 4096), (4096, 1))
    assert_size_stride(primals_175, (1024, ), (1, ))
    assert_size_stride(primals_176, (1024, ), (1, ))
    assert_size_stride(primals_177, (1024, ), (1, ))
    assert_size_stride(primals_178, (1024, ), (1, ))
    assert_size_stride(primals_179, (1024, ), (1, ))
    assert_size_stride(primals_180, (4096, 1024), (1024, 1))
    assert_size_stride(primals_181, (4096, ), (1, ))
    assert_size_stride(primals_182, (1024, 4096), (4096, 1))
    assert_size_stride(primals_183, (1024, ), (1, ))
    assert_size_stride(primals_184, (1024, ), (1, ))
    assert_size_stride(primals_185, (1024, ), (1, ))
    assert_size_stride(primals_186, (1024, ), (1, ))
    assert_size_stride(primals_187, (1024, ), (1, ))
    assert_size_stride(primals_188, (4096, 1024), (1024, 1))
    assert_size_stride(primals_189, (4096, ), (1, ))
    assert_size_stride(primals_190, (1024, 4096), (4096, 1))
    assert_size_stride(primals_191, (1024, ), (1, ))
    assert_size_stride(primals_192, (1024, ), (1, ))
    assert_size_stride(primals_193, (1024, ), (1, ))
    assert_size_stride(primals_194, (1024, ), (1, ))
    assert_size_stride(primals_195, (1024, ), (1, ))
    assert_size_stride(primals_196, (4096, 1024), (1024, 1))
    assert_size_stride(primals_197, (4096, ), (1, ))
    assert_size_stride(primals_198, (1024, 4096), (4096, 1))
    assert_size_stride(primals_199, (1024, ), (1, ))
    assert_size_stride(primals_200, (1024, ), (1, ))
    assert_size_stride(primals_201, (1024, ), (1, ))
    assert_size_stride(primals_202, (1024, ), (1, ))
    assert_size_stride(primals_203, (1024, ), (1, ))
    assert_size_stride(primals_204, (4096, 1024), (1024, 1))
    assert_size_stride(primals_205, (4096, ), (1, ))
    assert_size_stride(primals_206, (1024, 4096), (4096, 1))
    assert_size_stride(primals_207, (1024, ), (1, ))
    assert_size_stride(primals_208, (1024, ), (1, ))
    assert_size_stride(primals_209, (1024, ), (1, ))
    assert_size_stride(primals_210, (1024, ), (1, ))
    assert_size_stride(primals_211, (1024, ), (1, ))
    assert_size_stride(primals_212, (4096, 1024), (1024, 1))
    assert_size_stride(primals_213, (4096, ), (1, ))
    assert_size_stride(primals_214, (1024, 4096), (4096, 1))
    assert_size_stride(primals_215, (1024, ), (1, ))
    assert_size_stride(primals_216, (1024, ), (1, ))
    assert_size_stride(primals_217, (1024, ), (1, ))
    assert_size_stride(primals_218, (1024, ), (1, ))
    assert_size_stride(primals_219, (1024, ), (1, ))
    assert_size_stride(primals_220, (4096, 1024), (1024, 1))
    assert_size_stride(primals_221, (4096, ), (1, ))
    assert_size_stride(primals_222, (1024, 4096), (4096, 1))
    assert_size_stride(primals_223, (1024, ), (1, ))
    assert_size_stride(primals_224, (1024, ), (1, ))
    assert_size_stride(primals_225, (1024, ), (1, ))
    assert_size_stride(primals_226, (1024, ), (1, ))
    assert_size_stride(primals_227, (1024, ), (1, ))
    assert_size_stride(primals_228, (4096, 1024), (1024, 1))
    assert_size_stride(primals_229, (4096, ), (1, ))
    assert_size_stride(primals_230, (1024, 4096), (4096, 1))
    assert_size_stride(primals_231, (1024, ), (1, ))
    assert_size_stride(primals_232, (1024, ), (1, ))
    assert_size_stride(primals_233, (1024, ), (1, ))
    assert_size_stride(primals_234, (1024, ), (1, ))
    assert_size_stride(primals_235, (1024, ), (1, ))
    assert_size_stride(primals_236, (4096, 1024), (1024, 1))
    assert_size_stride(primals_237, (4096, ), (1, ))
    assert_size_stride(primals_238, (1024, 4096), (4096, 1))
    assert_size_stride(primals_239, (1024, ), (1, ))
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, ), (1, ))
    assert_size_stride(primals_242, (1024, ), (1, ))
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (4096, 1024), (1024, 1))
    assert_size_stride(primals_245, (4096, ), (1, ))
    assert_size_stride(primals_246, (1024, 4096), (4096, 1))
    assert_size_stride(primals_247, (1024, ), (1, ))
    assert_size_stride(primals_248, (1024, ), (1, ))
    assert_size_stride(primals_249, (1024, ), (1, ))
    assert_size_stride(primals_250, (1024, ), (1, ))
    assert_size_stride(primals_251, (1024, ), (1, ))
    assert_size_stride(primals_252, (4096, 1024), (1024, 1))
    assert_size_stride(primals_253, (4096, ), (1, ))
    assert_size_stride(primals_254, (1024, 4096), (4096, 1))
    assert_size_stride(primals_255, (1024, ), (1, ))
    assert_size_stride(primals_256, (1024, ), (1, ))
    assert_size_stride(primals_257, (1024, ), (1, ))
    assert_size_stride(primals_258, (1024, ), (1, ))
    assert_size_stride(primals_259, (1024, ), (1, ))
    assert_size_stride(primals_260, (4096, 1024), (1024, 1))
    assert_size_stride(primals_261, (4096, ), (1, ))
    assert_size_stride(primals_262, (1024, 4096), (4096, 1))
    assert_size_stride(primals_263, (1024, ), (1, ))
    assert_size_stride(primals_264, (1024, ), (1, ))
    assert_size_stride(primals_265, (1024, ), (1, ))
    assert_size_stride(primals_266, (1024, ), (1, ))
    assert_size_stride(primals_267, (1024, ), (1, ))
    assert_size_stride(primals_268, (4096, 1024), (1024, 1))
    assert_size_stride(primals_269, (4096, ), (1, ))
    assert_size_stride(primals_270, (1024, 4096), (4096, 1))
    assert_size_stride(primals_271, (1024, ), (1, ))
    assert_size_stride(primals_272, (1024, ), (1, ))
    assert_size_stride(primals_273, (1024, ), (1, ))
    assert_size_stride(primals_274, (1024, ), (1, ))
    assert_size_stride(primals_275, (1024, ), (1, ))
    assert_size_stride(primals_276, (4096, 1024), (1024, 1))
    assert_size_stride(primals_277, (4096, ), (1, ))
    assert_size_stride(primals_278, (1024, 4096), (4096, 1))
    assert_size_stride(primals_279, (1024, ), (1, ))
    assert_size_stride(primals_280, (1024, ), (1, ))
    assert_size_stride(primals_281, (1024, ), (1, ))
    assert_size_stride(primals_282, (1024, ), (1, ))
    assert_size_stride(primals_283, (1024, ), (1, ))
    assert_size_stride(primals_284, (4096, 1024), (1024, 1))
    assert_size_stride(primals_285, (4096, ), (1, ))
    assert_size_stride(primals_286, (1024, 4096), (4096, 1))
    assert_size_stride(primals_287, (1024, ), (1, ))
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_289, (1024, ), (1, ))
    assert_size_stride(primals_290, (1024, ), (1, ))
    assert_size_stride(primals_291, (1024, ), (1, ))
    assert_size_stride(primals_292, (4096, 1024), (1024, 1))
    assert_size_stride(primals_293, (4096, ), (1, ))
    assert_size_stride(primals_294, (1024, 4096), (4096, 1))
    assert_size_stride(primals_295, (1024, ), (1, ))
    assert_size_stride(primals_296, (1024, ), (1, ))
    assert_size_stride(primals_297, (1024, ), (1, ))
    assert_size_stride(primals_298, (1024, ), (1, ))
    assert_size_stride(primals_299, (1024, ), (1, ))
    assert_size_stride(primals_300, (4096, 1024), (1024, 1))
    assert_size_stride(primals_301, (4096, ), (1, ))
    assert_size_stride(primals_302, (1024, 4096), (4096, 1))
    assert_size_stride(primals_303, (1024, ), (1, ))
    assert_size_stride(primals_304, (1024, ), (1, ))
    assert_size_stride(primals_305, (1024, ), (1, ))
    assert_size_stride(primals_306, (1024, ), (1, ))
    assert_size_stride(primals_307, (1024, ), (1, ))
    assert_size_stride(primals_308, (4096, 1024), (1024, 1))
    assert_size_stride(primals_309, (4096, ), (1, ))
    assert_size_stride(primals_310, (1024, 4096), (4096, 1))
    assert_size_stride(primals_311, (1024, ), (1, ))
    assert_size_stride(primals_312, (1024, ), (1, ))
    assert_size_stride(primals_313, (1024, ), (1, ))
    assert_size_stride(primals_314, (1024, ), (1, ))
    assert_size_stride(primals_315, (1024, ), (1, ))
    assert_size_stride(primals_316, (4096, 1024), (1024, 1))
    assert_size_stride(primals_317, (4096, ), (1, ))
    assert_size_stride(primals_318, (1024, 4096), (4096, 1))
    assert_size_stride(primals_319, (1024, ), (1, ))
    assert_size_stride(primals_320, (1024, ), (1, ))
    assert_size_stride(primals_321, (1024, ), (1, ))
    assert_size_stride(primals_322, (1024, ), (1, ))
    assert_size_stride(primals_323, (1024, ), (1, ))
    assert_size_stride(primals_324, (4096, 1024), (1024, 1))
    assert_size_stride(primals_325, (4096, ), (1, ))
    assert_size_stride(primals_326, (1024, 4096), (4096, 1))
    assert_size_stride(primals_327, (1024, ), (1, ))
    assert_size_stride(primals_328, (1024, ), (1, ))
    assert_size_stride(primals_329, (1024, ), (1, ))
    assert_size_stride(primals_330, (1024, ), (1, ))
    assert_size_stride(primals_331, (1024, ), (1, ))
    assert_size_stride(primals_332, (4096, 1024), (1024, 1))
    assert_size_stride(primals_333, (4096, ), (1, ))
    assert_size_stride(primals_334, (1024, 4096), (4096, 1))
    assert_size_stride(primals_335, (1024, ), (1, ))
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_337, (1024, ), (1, ))
    assert_size_stride(primals_338, (1024, ), (1, ))
    assert_size_stride(primals_339, (1024, ), (1, ))
    assert_size_stride(primals_340, (4096, 1024), (1024, 1))
    assert_size_stride(primals_341, (4096, ), (1, ))
    assert_size_stride(primals_342, (1024, 4096), (4096, 1))
    assert_size_stride(primals_343, (1024, ), (1, ))
    assert_size_stride(primals_344, (1024, ), (1, ))
    assert_size_stride(primals_345, (1024, ), (1, ))
    assert_size_stride(primals_346, (1024, ), (1, ))
    assert_size_stride(primals_347, (1024, ), (1, ))
    assert_size_stride(primals_348, (4096, 1024), (1024, 1))
    assert_size_stride(primals_349, (4096, ), (1, ))
    assert_size_stride(primals_350, (1024, 4096), (4096, 1))
    assert_size_stride(primals_351, (1024, ), (1, ))
    assert_size_stride(primals_352, (1024, ), (1, ))
    assert_size_stride(primals_353, (1024, ), (1, ))
    assert_size_stride(primals_354, (1024, ), (1, ))
    assert_size_stride(primals_355, (1024, ), (1, ))
    assert_size_stride(primals_356, (4096, 1024), (1024, 1))
    assert_size_stride(primals_357, (4096, ), (1, ))
    assert_size_stride(primals_358, (1024, 4096), (4096, 1))
    assert_size_stride(primals_359, (1024, ), (1, ))
    assert_size_stride(primals_360, (1024, ), (1, ))
    assert_size_stride(primals_361, (1024, ), (1, ))
    assert_size_stride(primals_362, (32000, 1024), (1024, 1))
    assert_size_stride(primals_363, (32000, ), (1, ))
    assert_size_stride(primals_364, (1, 512), (512, 1))
    assert_size_stride(primals_365, (1, 512), (512, 1))
    buf0 = empty_strided((512, 1, 1024), (1024, 524288, 1), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_0(c_void_p(primals_364.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(buf0.data_ptr()))
    del primals_169
    # Source Nodes: [cat_1, word_emb_k], Original ATen: [aten.embedding, aten.native_dropout]
    buf1 = aten.native_dropout(buf0, 0.1, True)
    buf2 = buf1[0]
    buf3 = buf1[1]
    del buf1
    buf4 = empty((512, ), device='cpu', dtype=torch.int64)
    buf5 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_arange_cat_1(c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    # Source Nodes: [pos_emb_6], Original ATen: [aten.native_dropout]
    buf6 = aten.native_dropout(reinterpret_tensor(buf5, (1024, 1, 1024), (1024, 0, 1), 0), 0.1, True)
    buf7 = buf6[0]
    del buf6
    buf9 = reinterpret_tensor(buf0, (1, 512, 1024), (524288, 1024, 1), 0); del buf0  # reuse
    # Source Nodes: [q_head_h], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf2, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(primals_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf9)
    buf10 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf2, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(primals_2, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf10)
    buf11 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf2, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(primals_3, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf11)
    buf12 = reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf5  # reuse
    # Source Nodes: [k_head_r], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_4, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf12)
    del primals_4
    buf13 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf15 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_2(c_void_p(buf9.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf15.data_ptr()))
    del primals_5
    del primals_6
    buf14 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf13, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf10, (16, 64, 512), (64, 1, 1024), 0), out=buf14)
    buf16 = empty((16, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [bd], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf15, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf12, (16, 64, 1024), (64, 1, 1024), 0), out=buf16)
    buf17 = empty_strided((1, 16, 512, 1), (8192, 512, 1, 8192), device='cpu', dtype=torch.float32)
    buf18 = reinterpret_tensor(buf14, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf14  # reuse
    buf19 = empty_strided((1, 16, 512, 1), (8192, 512, 1, 8192), device='cpu', dtype=torch.float32)
    buf20 = buf18; del buf18  # reuse
    cpp_fused__softmax_add_index_select_mul_3(c_void_p(buf20.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf19.data_ptr()))
    # Source Nodes: [attn_prob, attn_prob_1], Original ATen: [aten._softmax, aten.native_dropout]
    buf21 = aten.native_dropout(buf20, 0.1, True)
    buf22 = buf21[0]
    buf23 = buf21[1]
    del buf21
    buf24 = reinterpret_tensor(buf9, (16, 512, 64), (32768, 64, 1), 0); del buf9  # reuse
    # Source Nodes: [attn_vec], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf22, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf11, (16, 512, 64), (64, 1024, 1), 0), out=buf24)
    buf25 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf26 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_4(c_void_p(buf24.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    del primals_7
    buf27 = reinterpret_tensor(buf24, (1, 512, 1024), (524288, 1024, 1), 0); del buf24  # reuse
    # Source Nodes: [attn_out], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf25, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf26, (1, 1024, 1024), (0, 1024, 1), 0), out=buf27)
    # Source Nodes: [attn_out_1], Original ATen: [aten.native_dropout]
    buf28 = aten.native_dropout(reinterpret_tensor(buf27, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf29 = buf28[0]
    buf30 = buf28[1]
    del buf28
    buf31 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf32 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf34 = reinterpret_tensor(buf27, (512, 1, 1024), (1024, 1024, 1), 0); del buf27  # reuse
    buf35 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_5(c_void_p(buf29.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()))
    buf36 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [output_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_173, buf35, reinterpret_tensor(primals_172, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf36)
    del primals_173
    buf37 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_6(c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    # Source Nodes: [output_3, output_4], Original ATen: [aten.gelu, aten.native_dropout]
    buf38 = aten.native_dropout(buf37, 0.1, True)
    buf39 = buf38[0]
    buf40 = buf38[1]
    del buf38
    buf41 = reinterpret_tensor(buf29, (512, 1024), (1024, 1), 0); del buf29  # reuse
    # Source Nodes: [output_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_175, reinterpret_tensor(buf39, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_174, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf41)
    del primals_175
    # Source Nodes: [output_6], Original ATen: [aten.native_dropout]
    buf42 = aten.native_dropout(reinterpret_tensor(buf41, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf43 = buf42[0]
    buf44 = buf42[1]
    del buf42
    buf45 = buf31; del buf31  # reuse
    buf46 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf48 = reinterpret_tensor(buf41, (512, 1, 1024), (1024, 1024, 1), 0); del buf41  # reuse
    buf49 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_7(c_void_p(buf43.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()))
    del primals_171
    del primals_177
    buf50 = reinterpret_tensor(buf43, (1, 512, 1024), (524288, 1024, 1), 0); del buf43  # reuse
    # Source Nodes: [q_head_h_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf49, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_8, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf50)
    buf51 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf49, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_9, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf51)
    buf52 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf49, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_10, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf52)
    buf53 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_11, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf53)
    del primals_11
    buf54 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf56 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_8(c_void_p(buf50.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()))
    del primals_12
    del primals_13
    buf55 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf54, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf51, (16, 64, 512), (64, 1, 1024), 0), out=buf55)
    buf57 = buf16; del buf16  # reuse
    # Source Nodes: [bd_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf56, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf53, (16, 64, 1024), (64, 1, 1024), 0), out=buf57)
    buf58 = buf19; del buf19  # reuse
    buf59 = reinterpret_tensor(buf55, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf55  # reuse
    buf60 = buf17; del buf17  # reuse
    buf61 = buf59; del buf59  # reuse
    cpp_fused__softmax_add_index_select_mul_9(c_void_p(buf61.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()))
    # Source Nodes: [attn_prob_2, attn_prob_3], Original ATen: [aten._softmax, aten.native_dropout]
    buf62 = aten.native_dropout(buf61, 0.1, True)
    buf63 = buf62[0]
    buf64 = buf62[1]
    del buf62
    buf65 = reinterpret_tensor(buf50, (16, 512, 64), (32768, 64, 1), 0); del buf50  # reuse
    # Source Nodes: [attn_vec_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf63, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf52, (16, 512, 64), (64, 1024, 1), 0), out=buf65)
    buf66 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf67 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_10(c_void_p(buf65.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    del primals_14
    buf68 = reinterpret_tensor(buf65, (1, 512, 1024), (524288, 1024, 1), 0); del buf65  # reuse
    # Source Nodes: [attn_out_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf66, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf67, (1, 1024, 1024), (0, 1024, 1), 0), out=buf68)
    # Source Nodes: [attn_out_4], Original ATen: [aten.native_dropout]
    buf69 = aten.native_dropout(reinterpret_tensor(buf68, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf70 = buf69[0]
    buf71 = buf69[1]
    del buf69
    buf72 = buf45; del buf45  # reuse
    buf73 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf75 = reinterpret_tensor(buf68, (512, 1, 1024), (1024, 1024, 1), 0); del buf68  # reuse
    buf76 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_11(c_void_p(buf70.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    buf77 = reinterpret_tensor(buf37, (512, 4096), (4096, 1), 0); del buf37  # reuse
    # Source Nodes: [output_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_181, buf76, reinterpret_tensor(primals_180, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf77)
    del primals_181
    buf78 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_12(c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()))
    # Source Nodes: [output_11, output_12], Original ATen: [aten.gelu, aten.native_dropout]
    buf79 = aten.native_dropout(buf78, 0.1, True)
    buf80 = buf79[0]
    buf81 = buf79[1]
    del buf79
    buf82 = reinterpret_tensor(buf70, (512, 1024), (1024, 1), 0); del buf70  # reuse
    # Source Nodes: [output_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_183, reinterpret_tensor(buf80, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_182, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf82)
    del primals_183
    # Source Nodes: [output_14], Original ATen: [aten.native_dropout]
    buf83 = aten.native_dropout(reinterpret_tensor(buf82, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf84 = buf83[0]
    buf85 = buf83[1]
    del buf83
    buf86 = buf72; del buf72  # reuse
    buf87 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf89 = reinterpret_tensor(buf82, (512, 1, 1024), (1024, 1024, 1), 0); del buf82  # reuse
    buf90 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_13(c_void_p(buf84.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del primals_179
    del primals_185
    buf91 = reinterpret_tensor(buf84, (1, 512, 1024), (524288, 1024, 1), 0); del buf84  # reuse
    # Source Nodes: [q_head_h_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf90, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_15, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf91)
    buf92 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf90, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_16, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf92)
    buf93 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf90, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_17, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf93)
    buf94 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_18, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf94)
    del primals_18
    buf95 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf97 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_14(c_void_p(buf91.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf97.data_ptr()))
    del primals_19
    del primals_20
    buf96 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf95, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf92, (16, 64, 512), (64, 1, 1024), 0), out=buf96)
    buf98 = buf57; del buf57  # reuse
    # Source Nodes: [bd_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf97, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf94, (16, 64, 1024), (64, 1, 1024), 0), out=buf98)
    buf99 = buf60; del buf60  # reuse
    buf100 = reinterpret_tensor(buf96, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf96  # reuse
    buf101 = buf58; del buf58  # reuse
    buf102 = buf100; del buf100  # reuse
    cpp_fused__softmax_add_index_select_mul_15(c_void_p(buf102.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf101.data_ptr()))
    # Source Nodes: [attn_prob_4, attn_prob_5], Original ATen: [aten._softmax, aten.native_dropout]
    buf103 = aten.native_dropout(buf102, 0.1, True)
    buf104 = buf103[0]
    buf105 = buf103[1]
    del buf103
    buf106 = reinterpret_tensor(buf91, (16, 512, 64), (32768, 64, 1), 0); del buf91  # reuse
    # Source Nodes: [attn_vec_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf104, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf93, (16, 512, 64), (64, 1024, 1), 0), out=buf106)
    buf107 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf108 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_16(c_void_p(buf106.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()))
    del primals_21
    buf109 = reinterpret_tensor(buf106, (1, 512, 1024), (524288, 1024, 1), 0); del buf106  # reuse
    # Source Nodes: [attn_out_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf107, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf108, (1, 1024, 1024), (0, 1024, 1), 0), out=buf109)
    # Source Nodes: [attn_out_7], Original ATen: [aten.native_dropout]
    buf110 = aten.native_dropout(reinterpret_tensor(buf109, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf111 = buf110[0]
    buf112 = buf110[1]
    del buf110
    buf113 = buf86; del buf86  # reuse
    buf114 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf116 = reinterpret_tensor(buf109, (512, 1, 1024), (1024, 1024, 1), 0); del buf109  # reuse
    buf117 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_17(c_void_p(buf111.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    buf118 = reinterpret_tensor(buf78, (512, 4096), (4096, 1), 0); del buf78  # reuse
    # Source Nodes: [output_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_189, buf117, reinterpret_tensor(primals_188, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf118)
    del primals_189
    buf119 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_18(c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()))
    # Source Nodes: [output_19, output_20], Original ATen: [aten.gelu, aten.native_dropout]
    buf120 = aten.native_dropout(buf119, 0.1, True)
    buf121 = buf120[0]
    buf122 = buf120[1]
    del buf120
    buf123 = reinterpret_tensor(buf111, (512, 1024), (1024, 1), 0); del buf111  # reuse
    # Source Nodes: [output_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_191, reinterpret_tensor(buf121, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_190, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf123)
    del primals_191
    # Source Nodes: [output_22], Original ATen: [aten.native_dropout]
    buf124 = aten.native_dropout(reinterpret_tensor(buf123, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf125 = buf124[0]
    buf126 = buf124[1]
    del buf124
    buf127 = buf113; del buf113  # reuse
    buf128 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf130 = reinterpret_tensor(buf123, (512, 1, 1024), (1024, 1024, 1), 0); del buf123  # reuse
    buf131 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_19(c_void_p(buf125.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    del primals_187
    del primals_193
    buf132 = reinterpret_tensor(buf125, (1, 512, 1024), (524288, 1024, 1), 0); del buf125  # reuse
    # Source Nodes: [q_head_h_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf131, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_22, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf132)
    buf133 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf131, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_23, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf133)
    buf134 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf131, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_24, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf134)
    buf135 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_25, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf135)
    del primals_25
    buf136 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf138 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_20(c_void_p(buf132.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()))
    del primals_26
    del primals_27
    buf137 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf136, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf133, (16, 64, 512), (64, 1, 1024), 0), out=buf137)
    buf139 = buf98; del buf98  # reuse
    # Source Nodes: [bd_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf138, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf135, (16, 64, 1024), (64, 1, 1024), 0), out=buf139)
    buf140 = buf99; del buf99  # reuse
    buf141 = reinterpret_tensor(buf137, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf137  # reuse
    buf142 = buf101; del buf101  # reuse
    buf143 = buf141; del buf141  # reuse
    cpp_fused__softmax_add_index_select_mul_21(c_void_p(buf143.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf142.data_ptr()))
    # Source Nodes: [attn_prob_6, attn_prob_7], Original ATen: [aten._softmax, aten.native_dropout]
    buf144 = aten.native_dropout(buf143, 0.1, True)
    buf145 = buf144[0]
    buf146 = buf144[1]
    del buf144
    buf147 = reinterpret_tensor(buf132, (16, 512, 64), (32768, 64, 1), 0); del buf132  # reuse
    # Source Nodes: [attn_vec_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf145, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf134, (16, 512, 64), (64, 1024, 1), 0), out=buf147)
    buf148 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf149 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_22(c_void_p(buf147.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()))
    del primals_28
    buf150 = reinterpret_tensor(buf147, (1, 512, 1024), (524288, 1024, 1), 0); del buf147  # reuse
    # Source Nodes: [attn_out_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf148, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf149, (1, 1024, 1024), (0, 1024, 1), 0), out=buf150)
    # Source Nodes: [attn_out_10], Original ATen: [aten.native_dropout]
    buf151 = aten.native_dropout(reinterpret_tensor(buf150, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf152 = buf151[0]
    buf153 = buf151[1]
    del buf151
    buf154 = buf127; del buf127  # reuse
    buf155 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf157 = reinterpret_tensor(buf150, (512, 1, 1024), (1024, 1024, 1), 0); del buf150  # reuse
    buf158 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_23(c_void_p(buf152.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    buf159 = reinterpret_tensor(buf119, (512, 4096), (4096, 1), 0); del buf119  # reuse
    # Source Nodes: [output_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_197, buf158, reinterpret_tensor(primals_196, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf159)
    del primals_197
    buf160 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_24(c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    # Source Nodes: [output_27, output_28], Original ATen: [aten.gelu, aten.native_dropout]
    buf161 = aten.native_dropout(buf160, 0.1, True)
    buf162 = buf161[0]
    buf163 = buf161[1]
    del buf161
    buf164 = reinterpret_tensor(buf152, (512, 1024), (1024, 1), 0); del buf152  # reuse
    # Source Nodes: [output_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_199, reinterpret_tensor(buf162, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_198, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf164)
    del primals_199
    # Source Nodes: [output_30], Original ATen: [aten.native_dropout]
    buf165 = aten.native_dropout(reinterpret_tensor(buf164, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf166 = buf165[0]
    buf167 = buf165[1]
    del buf165
    buf168 = buf154; del buf154  # reuse
    buf169 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf171 = reinterpret_tensor(buf164, (512, 1, 1024), (1024, 1024, 1), 0); del buf164  # reuse
    buf172 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_25(c_void_p(buf166.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del primals_195
    del primals_201
    buf173 = reinterpret_tensor(buf166, (1, 512, 1024), (524288, 1024, 1), 0); del buf166  # reuse
    # Source Nodes: [q_head_h_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf172, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_29, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf173)
    buf174 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf172, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_30, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf174)
    buf175 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf172, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_31, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf175)
    buf176 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_32, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf176)
    del primals_32
    buf177 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf179 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_26(c_void_p(buf173.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()))
    del primals_33
    del primals_34
    buf178 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf177, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf174, (16, 64, 512), (64, 1, 1024), 0), out=buf178)
    buf180 = buf139; del buf139  # reuse
    # Source Nodes: [bd_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf179, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf176, (16, 64, 1024), (64, 1, 1024), 0), out=buf180)
    buf181 = buf142; del buf142  # reuse
    buf182 = reinterpret_tensor(buf178, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf178  # reuse
    buf183 = buf140; del buf140  # reuse
    buf184 = buf182; del buf182  # reuse
    cpp_fused__softmax_add_index_select_mul_27(c_void_p(buf184.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()))
    # Source Nodes: [attn_prob_8, attn_prob_9], Original ATen: [aten._softmax, aten.native_dropout]
    buf185 = aten.native_dropout(buf184, 0.1, True)
    buf186 = buf185[0]
    buf187 = buf185[1]
    del buf185
    buf188 = reinterpret_tensor(buf173, (16, 512, 64), (32768, 64, 1), 0); del buf173  # reuse
    # Source Nodes: [attn_vec_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf186, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf175, (16, 512, 64), (64, 1024, 1), 0), out=buf188)
    buf189 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf190 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_28(c_void_p(buf188.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    del primals_35
    buf191 = reinterpret_tensor(buf188, (1, 512, 1024), (524288, 1024, 1), 0); del buf188  # reuse
    # Source Nodes: [attn_out_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf189, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf190, (1, 1024, 1024), (0, 1024, 1), 0), out=buf191)
    # Source Nodes: [attn_out_13], Original ATen: [aten.native_dropout]
    buf192 = aten.native_dropout(reinterpret_tensor(buf191, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf193 = buf192[0]
    buf194 = buf192[1]
    del buf192
    buf195 = buf168; del buf168  # reuse
    buf196 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf198 = reinterpret_tensor(buf191, (512, 1, 1024), (1024, 1024, 1), 0); del buf191  # reuse
    buf199 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_29(c_void_p(buf193.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    buf200 = reinterpret_tensor(buf160, (512, 4096), (4096, 1), 0); del buf160  # reuse
    # Source Nodes: [output_34], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_205, buf199, reinterpret_tensor(primals_204, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf200)
    del primals_205
    buf201 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_30(c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    # Source Nodes: [output_35, output_36], Original ATen: [aten.gelu, aten.native_dropout]
    buf202 = aten.native_dropout(buf201, 0.1, True)
    buf203 = buf202[0]
    buf204 = buf202[1]
    del buf202
    buf205 = reinterpret_tensor(buf193, (512, 1024), (1024, 1), 0); del buf193  # reuse
    # Source Nodes: [output_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_207, reinterpret_tensor(buf203, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_206, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf205)
    del primals_207
    # Source Nodes: [output_38], Original ATen: [aten.native_dropout]
    buf206 = aten.native_dropout(reinterpret_tensor(buf205, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf207 = buf206[0]
    buf208 = buf206[1]
    del buf206
    buf209 = buf195; del buf195  # reuse
    buf210 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf212 = reinterpret_tensor(buf205, (512, 1, 1024), (1024, 1024, 1), 0); del buf205  # reuse
    buf213 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_31(c_void_p(buf207.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    del primals_203
    del primals_209
    buf214 = reinterpret_tensor(buf207, (1, 512, 1024), (524288, 1024, 1), 0); del buf207  # reuse
    # Source Nodes: [q_head_h_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf213, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_36, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf214)
    buf215 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf213, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_37, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf215)
    buf216 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf213, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_38, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf216)
    buf217 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_39, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf217)
    del primals_39
    buf218 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf220 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_32(c_void_p(buf214.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()))
    del primals_40
    del primals_41
    buf219 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf218, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf215, (16, 64, 512), (64, 1, 1024), 0), out=buf219)
    buf221 = buf180; del buf180  # reuse
    # Source Nodes: [bd_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf220, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf217, (16, 64, 1024), (64, 1, 1024), 0), out=buf221)
    buf222 = buf183; del buf183  # reuse
    buf223 = reinterpret_tensor(buf219, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf219  # reuse
    buf224 = buf181; del buf181  # reuse
    buf225 = buf223; del buf223  # reuse
    cpp_fused__softmax_add_index_select_mul_33(c_void_p(buf225.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf224.data_ptr()))
    # Source Nodes: [attn_prob_10, attn_prob_11], Original ATen: [aten._softmax, aten.native_dropout]
    buf226 = aten.native_dropout(buf225, 0.1, True)
    buf227 = buf226[0]
    buf228 = buf226[1]
    del buf226
    buf229 = reinterpret_tensor(buf214, (16, 512, 64), (32768, 64, 1), 0); del buf214  # reuse
    # Source Nodes: [attn_vec_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf227, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf216, (16, 512, 64), (64, 1024, 1), 0), out=buf229)
    buf230 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf231 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_34(c_void_p(buf229.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del primals_42
    buf232 = reinterpret_tensor(buf229, (1, 512, 1024), (524288, 1024, 1), 0); del buf229  # reuse
    # Source Nodes: [attn_out_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf230, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf231, (1, 1024, 1024), (0, 1024, 1), 0), out=buf232)
    # Source Nodes: [attn_out_16], Original ATen: [aten.native_dropout]
    buf233 = aten.native_dropout(reinterpret_tensor(buf232, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf234 = buf233[0]
    buf235 = buf233[1]
    del buf233
    buf236 = buf209; del buf209  # reuse
    buf237 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf239 = reinterpret_tensor(buf232, (512, 1, 1024), (1024, 1024, 1), 0); del buf232  # reuse
    buf240 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_35(c_void_p(buf234.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = reinterpret_tensor(buf201, (512, 4096), (4096, 1), 0); del buf201  # reuse
    # Source Nodes: [output_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_213, buf240, reinterpret_tensor(primals_212, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf241)
    del primals_213
    buf242 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_36(c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    # Source Nodes: [output_43, output_44], Original ATen: [aten.gelu, aten.native_dropout]
    buf243 = aten.native_dropout(buf242, 0.1, True)
    buf244 = buf243[0]
    buf245 = buf243[1]
    del buf243
    buf246 = reinterpret_tensor(buf234, (512, 1024), (1024, 1), 0); del buf234  # reuse
    # Source Nodes: [output_45], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_215, reinterpret_tensor(buf244, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_214, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf246)
    del primals_215
    # Source Nodes: [output_46], Original ATen: [aten.native_dropout]
    buf247 = aten.native_dropout(reinterpret_tensor(buf246, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf248 = buf247[0]
    buf249 = buf247[1]
    del buf247
    buf250 = buf236; del buf236  # reuse
    buf251 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf253 = reinterpret_tensor(buf246, (512, 1, 1024), (1024, 1024, 1), 0); del buf246  # reuse
    buf254 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_37(c_void_p(buf248.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    del primals_211
    del primals_217
    buf255 = reinterpret_tensor(buf248, (1, 512, 1024), (524288, 1024, 1), 0); del buf248  # reuse
    # Source Nodes: [q_head_h_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf254, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_43, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf255)
    buf256 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf254, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_44, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf256)
    buf257 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf254, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_45, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf257)
    buf258 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_46, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf258)
    del primals_46
    buf259 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf261 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_38(c_void_p(buf255.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()))
    del primals_47
    del primals_48
    buf260 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf259, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf256, (16, 64, 512), (64, 1, 1024), 0), out=buf260)
    buf262 = buf221; del buf221  # reuse
    # Source Nodes: [bd_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf261, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf258, (16, 64, 1024), (64, 1, 1024), 0), out=buf262)
    buf263 = buf224; del buf224  # reuse
    buf264 = reinterpret_tensor(buf260, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf260  # reuse
    buf265 = buf222; del buf222  # reuse
    buf266 = buf264; del buf264  # reuse
    cpp_fused__softmax_add_index_select_mul_39(c_void_p(buf266.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf265.data_ptr()))
    # Source Nodes: [attn_prob_12, attn_prob_13], Original ATen: [aten._softmax, aten.native_dropout]
    buf267 = aten.native_dropout(buf266, 0.1, True)
    buf268 = buf267[0]
    buf269 = buf267[1]
    del buf267
    buf270 = reinterpret_tensor(buf255, (16, 512, 64), (32768, 64, 1), 0); del buf255  # reuse
    # Source Nodes: [attn_vec_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf268, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf257, (16, 512, 64), (64, 1024, 1), 0), out=buf270)
    buf271 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf272 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_40(c_void_p(buf270.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()))
    del primals_49
    buf273 = reinterpret_tensor(buf270, (1, 512, 1024), (524288, 1024, 1), 0); del buf270  # reuse
    # Source Nodes: [attn_out_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf271, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf272, (1, 1024, 1024), (0, 1024, 1), 0), out=buf273)
    # Source Nodes: [attn_out_19], Original ATen: [aten.native_dropout]
    buf274 = aten.native_dropout(reinterpret_tensor(buf273, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf275 = buf274[0]
    buf276 = buf274[1]
    del buf274
    buf277 = buf250; del buf250  # reuse
    buf278 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf280 = reinterpret_tensor(buf273, (512, 1, 1024), (1024, 1024, 1), 0); del buf273  # reuse
    buf281 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_41(c_void_p(buf275.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    buf282 = reinterpret_tensor(buf242, (512, 4096), (4096, 1), 0); del buf242  # reuse
    # Source Nodes: [output_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_221, buf281, reinterpret_tensor(primals_220, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf282)
    del primals_221
    buf283 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_42(c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()))
    # Source Nodes: [output_51, output_52], Original ATen: [aten.gelu, aten.native_dropout]
    buf284 = aten.native_dropout(buf283, 0.1, True)
    buf285 = buf284[0]
    buf286 = buf284[1]
    del buf284
    buf287 = reinterpret_tensor(buf275, (512, 1024), (1024, 1), 0); del buf275  # reuse
    # Source Nodes: [output_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_223, reinterpret_tensor(buf285, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_222, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf287)
    del primals_223
    # Source Nodes: [output_54], Original ATen: [aten.native_dropout]
    buf288 = aten.native_dropout(reinterpret_tensor(buf287, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf289 = buf288[0]
    buf290 = buf288[1]
    del buf288
    buf291 = buf277; del buf277  # reuse
    buf292 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf294 = reinterpret_tensor(buf287, (512, 1, 1024), (1024, 1024, 1), 0); del buf287  # reuse
    buf295 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_43(c_void_p(buf289.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()))
    del primals_219
    del primals_225
    buf296 = reinterpret_tensor(buf289, (1, 512, 1024), (524288, 1024, 1), 0); del buf289  # reuse
    # Source Nodes: [q_head_h_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf295, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_50, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf296)
    buf297 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf295, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_51, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf297)
    buf298 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf295, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_52, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf298)
    buf299 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_53, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf299)
    del primals_53
    buf300 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf302 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_44(c_void_p(buf296.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf302.data_ptr()))
    del primals_54
    del primals_55
    buf301 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf300, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf297, (16, 64, 512), (64, 1, 1024), 0), out=buf301)
    buf303 = buf262; del buf262  # reuse
    # Source Nodes: [bd_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf302, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf299, (16, 64, 1024), (64, 1, 1024), 0), out=buf303)
    buf304 = buf265; del buf265  # reuse
    buf305 = reinterpret_tensor(buf301, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf301  # reuse
    buf306 = buf263; del buf263  # reuse
    buf307 = buf305; del buf305  # reuse
    cpp_fused__softmax_add_index_select_mul_45(c_void_p(buf307.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()))
    # Source Nodes: [attn_prob_14, attn_prob_15], Original ATen: [aten._softmax, aten.native_dropout]
    buf308 = aten.native_dropout(buf307, 0.1, True)
    buf309 = buf308[0]
    buf310 = buf308[1]
    del buf308
    buf311 = reinterpret_tensor(buf296, (16, 512, 64), (32768, 64, 1), 0); del buf296  # reuse
    # Source Nodes: [attn_vec_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf309, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf298, (16, 512, 64), (64, 1024, 1), 0), out=buf311)
    buf312 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf313 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_46(c_void_p(buf311.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()))
    del primals_56
    buf314 = reinterpret_tensor(buf311, (1, 512, 1024), (524288, 1024, 1), 0); del buf311  # reuse
    # Source Nodes: [attn_out_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf312, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf313, (1, 1024, 1024), (0, 1024, 1), 0), out=buf314)
    # Source Nodes: [attn_out_22], Original ATen: [aten.native_dropout]
    buf315 = aten.native_dropout(reinterpret_tensor(buf314, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf316 = buf315[0]
    buf317 = buf315[1]
    del buf315
    buf318 = buf291; del buf291  # reuse
    buf319 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf321 = reinterpret_tensor(buf314, (512, 1, 1024), (1024, 1024, 1), 0); del buf314  # reuse
    buf322 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_47(c_void_p(buf316.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()))
    buf323 = reinterpret_tensor(buf283, (512, 4096), (4096, 1), 0); del buf283  # reuse
    # Source Nodes: [output_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_229, buf322, reinterpret_tensor(primals_228, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf323)
    del primals_229
    buf324 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_48(c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()))
    # Source Nodes: [output_59, output_60], Original ATen: [aten.gelu, aten.native_dropout]
    buf325 = aten.native_dropout(buf324, 0.1, True)
    buf326 = buf325[0]
    buf327 = buf325[1]
    del buf325
    buf328 = reinterpret_tensor(buf316, (512, 1024), (1024, 1), 0); del buf316  # reuse
    # Source Nodes: [output_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_231, reinterpret_tensor(buf326, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_230, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf328)
    del primals_231
    # Source Nodes: [output_62], Original ATen: [aten.native_dropout]
    buf329 = aten.native_dropout(reinterpret_tensor(buf328, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf330 = buf329[0]
    buf331 = buf329[1]
    del buf329
    buf332 = buf318; del buf318  # reuse
    buf333 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf335 = reinterpret_tensor(buf328, (512, 1, 1024), (1024, 1024, 1), 0); del buf328  # reuse
    buf336 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_49(c_void_p(buf330.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    del primals_227
    del primals_233
    buf337 = reinterpret_tensor(buf330, (1, 512, 1024), (524288, 1024, 1), 0); del buf330  # reuse
    # Source Nodes: [q_head_h_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf336, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_57, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf337)
    buf338 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf336, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_58, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf338)
    buf339 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf336, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_59, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf339)
    buf340 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_60, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf340)
    del primals_60
    buf341 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf343 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_50(c_void_p(buf337.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf343.data_ptr()))
    del primals_61
    del primals_62
    buf342 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf341, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf338, (16, 64, 512), (64, 1, 1024), 0), out=buf342)
    buf344 = buf303; del buf303  # reuse
    # Source Nodes: [bd_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf343, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf340, (16, 64, 1024), (64, 1, 1024), 0), out=buf344)
    buf345 = buf306; del buf306  # reuse
    buf346 = reinterpret_tensor(buf342, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf342  # reuse
    buf347 = buf304; del buf304  # reuse
    buf348 = buf346; del buf346  # reuse
    cpp_fused__softmax_add_index_select_mul_51(c_void_p(buf348.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()))
    # Source Nodes: [attn_prob_16, attn_prob_17], Original ATen: [aten._softmax, aten.native_dropout]
    buf349 = aten.native_dropout(buf348, 0.1, True)
    buf350 = buf349[0]
    buf351 = buf349[1]
    del buf349
    buf352 = reinterpret_tensor(buf337, (16, 512, 64), (32768, 64, 1), 0); del buf337  # reuse
    # Source Nodes: [attn_vec_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf350, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf339, (16, 512, 64), (64, 1024, 1), 0), out=buf352)
    buf353 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf354 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_52(c_void_p(buf352.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()))
    del primals_63
    buf355 = reinterpret_tensor(buf352, (1, 512, 1024), (524288, 1024, 1), 0); del buf352  # reuse
    # Source Nodes: [attn_out_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf353, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf354, (1, 1024, 1024), (0, 1024, 1), 0), out=buf355)
    # Source Nodes: [attn_out_25], Original ATen: [aten.native_dropout]
    buf356 = aten.native_dropout(reinterpret_tensor(buf355, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf357 = buf356[0]
    buf358 = buf356[1]
    del buf356
    buf359 = buf332; del buf332  # reuse
    buf360 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf362 = reinterpret_tensor(buf355, (512, 1, 1024), (1024, 1024, 1), 0); del buf355  # reuse
    buf363 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_53(c_void_p(buf357.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()))
    buf364 = reinterpret_tensor(buf324, (512, 4096), (4096, 1), 0); del buf324  # reuse
    # Source Nodes: [output_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_237, buf363, reinterpret_tensor(primals_236, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf364)
    del primals_237
    buf365 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_54(c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()))
    # Source Nodes: [output_67, output_68], Original ATen: [aten.gelu, aten.native_dropout]
    buf366 = aten.native_dropout(buf365, 0.1, True)
    buf367 = buf366[0]
    buf368 = buf366[1]
    del buf366
    buf369 = reinterpret_tensor(buf357, (512, 1024), (1024, 1), 0); del buf357  # reuse
    # Source Nodes: [output_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_239, reinterpret_tensor(buf367, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_238, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf369)
    del primals_239
    # Source Nodes: [output_70], Original ATen: [aten.native_dropout]
    buf370 = aten.native_dropout(reinterpret_tensor(buf369, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf371 = buf370[0]
    buf372 = buf370[1]
    del buf370
    buf373 = buf359; del buf359  # reuse
    buf374 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf376 = reinterpret_tensor(buf369, (512, 1, 1024), (1024, 1024, 1), 0); del buf369  # reuse
    buf377 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_55(c_void_p(buf371.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    del primals_235
    del primals_241
    buf378 = reinterpret_tensor(buf371, (1, 512, 1024), (524288, 1024, 1), 0); del buf371  # reuse
    # Source Nodes: [q_head_h_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf377, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_64, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf378)
    buf379 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf377, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_65, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf379)
    buf380 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf377, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_66, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf380)
    buf381 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_67, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf381)
    del primals_67
    buf382 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf384 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_56(c_void_p(buf378.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf384.data_ptr()))
    del primals_68
    del primals_69
    buf383 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf382, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf379, (16, 64, 512), (64, 1, 1024), 0), out=buf383)
    buf385 = buf344; del buf344  # reuse
    # Source Nodes: [bd_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf384, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf381, (16, 64, 1024), (64, 1, 1024), 0), out=buf385)
    buf386 = buf347; del buf347  # reuse
    buf387 = reinterpret_tensor(buf383, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf383  # reuse
    buf388 = buf345; del buf345  # reuse
    buf389 = buf387; del buf387  # reuse
    cpp_fused__softmax_add_index_select_mul_57(c_void_p(buf389.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf388.data_ptr()))
    # Source Nodes: [attn_prob_18, attn_prob_19], Original ATen: [aten._softmax, aten.native_dropout]
    buf390 = aten.native_dropout(buf389, 0.1, True)
    buf391 = buf390[0]
    buf392 = buf390[1]
    del buf390
    buf393 = reinterpret_tensor(buf378, (16, 512, 64), (32768, 64, 1), 0); del buf378  # reuse
    # Source Nodes: [attn_vec_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf391, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf380, (16, 512, 64), (64, 1024, 1), 0), out=buf393)
    buf394 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf395 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_58(c_void_p(buf393.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()))
    del primals_70
    buf396 = reinterpret_tensor(buf393, (1, 512, 1024), (524288, 1024, 1), 0); del buf393  # reuse
    # Source Nodes: [attn_out_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf394, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf395, (1, 1024, 1024), (0, 1024, 1), 0), out=buf396)
    # Source Nodes: [attn_out_28], Original ATen: [aten.native_dropout]
    buf397 = aten.native_dropout(reinterpret_tensor(buf396, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf398 = buf397[0]
    buf399 = buf397[1]
    del buf397
    buf400 = buf373; del buf373  # reuse
    buf401 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf403 = reinterpret_tensor(buf396, (512, 1, 1024), (1024, 1024, 1), 0); del buf396  # reuse
    buf404 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_59(c_void_p(buf398.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()))
    buf405 = reinterpret_tensor(buf365, (512, 4096), (4096, 1), 0); del buf365  # reuse
    # Source Nodes: [output_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_245, buf404, reinterpret_tensor(primals_244, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf405)
    del primals_245
    buf406 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_60(c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()))
    # Source Nodes: [output_75, output_76], Original ATen: [aten.gelu, aten.native_dropout]
    buf407 = aten.native_dropout(buf406, 0.1, True)
    buf408 = buf407[0]
    buf409 = buf407[1]
    del buf407
    buf410 = reinterpret_tensor(buf398, (512, 1024), (1024, 1), 0); del buf398  # reuse
    # Source Nodes: [output_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_247, reinterpret_tensor(buf408, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_246, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf410)
    del primals_247
    # Source Nodes: [output_78], Original ATen: [aten.native_dropout]
    buf411 = aten.native_dropout(reinterpret_tensor(buf410, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf412 = buf411[0]
    buf413 = buf411[1]
    del buf411
    buf414 = buf400; del buf400  # reuse
    buf415 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf417 = reinterpret_tensor(buf410, (512, 1, 1024), (1024, 1024, 1), 0); del buf410  # reuse
    buf418 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_61(c_void_p(buf412.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()))
    del primals_243
    del primals_249
    buf419 = reinterpret_tensor(buf412, (1, 512, 1024), (524288, 1024, 1), 0); del buf412  # reuse
    # Source Nodes: [q_head_h_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf418, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_71, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf419)
    buf420 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf418, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_72, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf420)
    buf421 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf418, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_73, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf421)
    buf422 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_74, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf422)
    del primals_74
    buf423 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf425 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_62(c_void_p(buf419.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf425.data_ptr()))
    del primals_75
    del primals_76
    buf424 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf423, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf420, (16, 64, 512), (64, 1, 1024), 0), out=buf424)
    buf426 = buf385; del buf385  # reuse
    # Source Nodes: [bd_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf425, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf422, (16, 64, 1024), (64, 1, 1024), 0), out=buf426)
    buf427 = buf388; del buf388  # reuse
    buf428 = reinterpret_tensor(buf424, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf424  # reuse
    buf429 = buf386; del buf386  # reuse
    buf430 = buf428; del buf428  # reuse
    cpp_fused__softmax_add_index_select_mul_63(c_void_p(buf430.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf429.data_ptr()))
    # Source Nodes: [attn_prob_20, attn_prob_21], Original ATen: [aten._softmax, aten.native_dropout]
    buf431 = aten.native_dropout(buf430, 0.1, True)
    buf432 = buf431[0]
    buf433 = buf431[1]
    del buf431
    buf434 = reinterpret_tensor(buf419, (16, 512, 64), (32768, 64, 1), 0); del buf419  # reuse
    # Source Nodes: [attn_vec_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf432, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf421, (16, 512, 64), (64, 1024, 1), 0), out=buf434)
    buf435 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf436 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_64(c_void_p(buf434.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()))
    del primals_77
    buf437 = reinterpret_tensor(buf434, (1, 512, 1024), (524288, 1024, 1), 0); del buf434  # reuse
    # Source Nodes: [attn_out_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf435, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf436, (1, 1024, 1024), (0, 1024, 1), 0), out=buf437)
    # Source Nodes: [attn_out_31], Original ATen: [aten.native_dropout]
    buf438 = aten.native_dropout(reinterpret_tensor(buf437, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf439 = buf438[0]
    buf440 = buf438[1]
    del buf438
    buf441 = buf414; del buf414  # reuse
    buf442 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf444 = reinterpret_tensor(buf437, (512, 1, 1024), (1024, 1024, 1), 0); del buf437  # reuse
    buf445 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_65(c_void_p(buf439.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()))
    buf446 = reinterpret_tensor(buf406, (512, 4096), (4096, 1), 0); del buf406  # reuse
    # Source Nodes: [output_82], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_253, buf445, reinterpret_tensor(primals_252, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf446)
    del primals_253
    buf447 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_66(c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()))
    # Source Nodes: [output_83, output_84], Original ATen: [aten.gelu, aten.native_dropout]
    buf448 = aten.native_dropout(buf447, 0.1, True)
    buf449 = buf448[0]
    buf450 = buf448[1]
    del buf448
    buf451 = reinterpret_tensor(buf439, (512, 1024), (1024, 1), 0); del buf439  # reuse
    # Source Nodes: [output_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_255, reinterpret_tensor(buf449, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_254, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf451)
    del primals_255
    # Source Nodes: [output_86], Original ATen: [aten.native_dropout]
    buf452 = aten.native_dropout(reinterpret_tensor(buf451, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf453 = buf452[0]
    buf454 = buf452[1]
    del buf452
    buf455 = buf441; del buf441  # reuse
    buf456 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf458 = reinterpret_tensor(buf451, (512, 1, 1024), (1024, 1024, 1), 0); del buf451  # reuse
    buf459 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_67(c_void_p(buf453.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()))
    del primals_251
    del primals_257
    buf460 = reinterpret_tensor(buf453, (1, 512, 1024), (524288, 1024, 1), 0); del buf453  # reuse
    # Source Nodes: [q_head_h_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf459, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_78, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf460)
    buf461 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf459, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_79, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf461)
    buf462 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf459, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_80, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf462)
    buf463 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_81, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf463)
    del primals_81
    buf464 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf466 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_68(c_void_p(buf460.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf466.data_ptr()))
    del primals_82
    del primals_83
    buf465 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf464, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf461, (16, 64, 512), (64, 1, 1024), 0), out=buf465)
    buf467 = buf426; del buf426  # reuse
    # Source Nodes: [bd_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf466, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf463, (16, 64, 1024), (64, 1, 1024), 0), out=buf467)
    buf468 = buf429; del buf429  # reuse
    buf469 = reinterpret_tensor(buf465, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf465  # reuse
    buf470 = buf427; del buf427  # reuse
    buf471 = buf469; del buf469  # reuse
    cpp_fused__softmax_add_index_select_mul_69(c_void_p(buf471.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf470.data_ptr()))
    # Source Nodes: [attn_prob_22, attn_prob_23], Original ATen: [aten._softmax, aten.native_dropout]
    buf472 = aten.native_dropout(buf471, 0.1, True)
    buf473 = buf472[0]
    buf474 = buf472[1]
    del buf472
    buf475 = reinterpret_tensor(buf460, (16, 512, 64), (32768, 64, 1), 0); del buf460  # reuse
    # Source Nodes: [attn_vec_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf473, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf462, (16, 512, 64), (64, 1024, 1), 0), out=buf475)
    buf476 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf477 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_70(c_void_p(buf475.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()))
    del primals_84
    buf478 = reinterpret_tensor(buf475, (1, 512, 1024), (524288, 1024, 1), 0); del buf475  # reuse
    # Source Nodes: [attn_out_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf476, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf477, (1, 1024, 1024), (0, 1024, 1), 0), out=buf478)
    # Source Nodes: [attn_out_34], Original ATen: [aten.native_dropout]
    buf479 = aten.native_dropout(reinterpret_tensor(buf478, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf480 = buf479[0]
    buf481 = buf479[1]
    del buf479
    buf482 = buf455; del buf455  # reuse
    buf483 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf485 = reinterpret_tensor(buf478, (512, 1, 1024), (1024, 1024, 1), 0); del buf478  # reuse
    buf486 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_71(c_void_p(buf480.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()))
    buf487 = reinterpret_tensor(buf447, (512, 4096), (4096, 1), 0); del buf447  # reuse
    # Source Nodes: [output_90], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_261, buf486, reinterpret_tensor(primals_260, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf487)
    del primals_261
    buf488 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_72(c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()))
    # Source Nodes: [output_91, output_92], Original ATen: [aten.gelu, aten.native_dropout]
    buf489 = aten.native_dropout(buf488, 0.1, True)
    buf490 = buf489[0]
    buf491 = buf489[1]
    del buf489
    buf492 = reinterpret_tensor(buf480, (512, 1024), (1024, 1), 0); del buf480  # reuse
    # Source Nodes: [output_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_263, reinterpret_tensor(buf490, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_262, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf492)
    del primals_263
    # Source Nodes: [output_94], Original ATen: [aten.native_dropout]
    buf493 = aten.native_dropout(reinterpret_tensor(buf492, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf494 = buf493[0]
    buf495 = buf493[1]
    del buf493
    buf496 = buf482; del buf482  # reuse
    buf497 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf499 = reinterpret_tensor(buf492, (512, 1, 1024), (1024, 1024, 1), 0); del buf492  # reuse
    buf500 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_73(c_void_p(buf494.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()))
    del primals_259
    del primals_265
    buf501 = reinterpret_tensor(buf494, (1, 512, 1024), (524288, 1024, 1), 0); del buf494  # reuse
    # Source Nodes: [q_head_h_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf500, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_85, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf501)
    buf502 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf500, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_86, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf502)
    buf503 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf500, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_87, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf503)
    buf504 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_88, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf504)
    del primals_88
    buf505 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf507 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_74(c_void_p(buf501.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf507.data_ptr()))
    del primals_89
    del primals_90
    buf506 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf505, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf502, (16, 64, 512), (64, 1, 1024), 0), out=buf506)
    buf508 = buf467; del buf467  # reuse
    # Source Nodes: [bd_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf507, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf504, (16, 64, 1024), (64, 1, 1024), 0), out=buf508)
    buf509 = buf470; del buf470  # reuse
    buf510 = reinterpret_tensor(buf506, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf506  # reuse
    buf511 = buf468; del buf468  # reuse
    buf512 = buf510; del buf510  # reuse
    cpp_fused__softmax_add_index_select_mul_75(c_void_p(buf512.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf511.data_ptr()))
    # Source Nodes: [attn_prob_24, attn_prob_25], Original ATen: [aten._softmax, aten.native_dropout]
    buf513 = aten.native_dropout(buf512, 0.1, True)
    buf514 = buf513[0]
    buf515 = buf513[1]
    del buf513
    buf516 = reinterpret_tensor(buf501, (16, 512, 64), (32768, 64, 1), 0); del buf501  # reuse
    # Source Nodes: [attn_vec_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf514, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf503, (16, 512, 64), (64, 1024, 1), 0), out=buf516)
    buf517 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf518 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_76(c_void_p(buf516.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf518.data_ptr()))
    del primals_91
    buf519 = reinterpret_tensor(buf516, (1, 512, 1024), (524288, 1024, 1), 0); del buf516  # reuse
    # Source Nodes: [attn_out_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf517, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf518, (1, 1024, 1024), (0, 1024, 1), 0), out=buf519)
    # Source Nodes: [attn_out_37], Original ATen: [aten.native_dropout]
    buf520 = aten.native_dropout(reinterpret_tensor(buf519, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf521 = buf520[0]
    buf522 = buf520[1]
    del buf520
    buf523 = buf496; del buf496  # reuse
    buf524 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf526 = reinterpret_tensor(buf519, (512, 1, 1024), (1024, 1024, 1), 0); del buf519  # reuse
    buf527 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_77(c_void_p(buf521.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf527.data_ptr()))
    buf528 = reinterpret_tensor(buf488, (512, 4096), (4096, 1), 0); del buf488  # reuse
    # Source Nodes: [output_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_269, buf527, reinterpret_tensor(primals_268, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf528)
    del primals_269
    buf529 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_78(c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()))
    # Source Nodes: [output_100, output_99], Original ATen: [aten.gelu, aten.native_dropout]
    buf530 = aten.native_dropout(buf529, 0.1, True)
    buf531 = buf530[0]
    buf532 = buf530[1]
    del buf530
    buf533 = reinterpret_tensor(buf521, (512, 1024), (1024, 1), 0); del buf521  # reuse
    # Source Nodes: [output_101], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_271, reinterpret_tensor(buf531, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_270, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf533)
    del primals_271
    # Source Nodes: [output_102], Original ATen: [aten.native_dropout]
    buf534 = aten.native_dropout(reinterpret_tensor(buf533, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf535 = buf534[0]
    buf536 = buf534[1]
    del buf534
    buf537 = buf523; del buf523  # reuse
    buf538 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf540 = reinterpret_tensor(buf533, (512, 1, 1024), (1024, 1024, 1), 0); del buf533  # reuse
    buf541 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_79(c_void_p(buf535.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()))
    del primals_267
    del primals_273
    buf542 = reinterpret_tensor(buf535, (1, 512, 1024), (524288, 1024, 1), 0); del buf535  # reuse
    # Source Nodes: [q_head_h_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf541, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_92, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf542)
    buf543 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf541, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_93, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf543)
    buf544 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf541, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_94, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf544)
    buf545 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_95, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf545)
    del primals_95
    buf546 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf548 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_80(c_void_p(buf542.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf548.data_ptr()))
    del primals_96
    del primals_97
    buf547 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf546, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf543, (16, 64, 512), (64, 1, 1024), 0), out=buf547)
    buf549 = buf508; del buf508  # reuse
    # Source Nodes: [bd_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf548, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf545, (16, 64, 1024), (64, 1, 1024), 0), out=buf549)
    buf550 = buf511; del buf511  # reuse
    buf551 = reinterpret_tensor(buf547, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf547  # reuse
    buf552 = buf509; del buf509  # reuse
    buf553 = buf551; del buf551  # reuse
    cpp_fused__softmax_add_index_select_mul_81(c_void_p(buf553.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf552.data_ptr()))
    # Source Nodes: [attn_prob_26, attn_prob_27], Original ATen: [aten._softmax, aten.native_dropout]
    buf554 = aten.native_dropout(buf553, 0.1, True)
    buf555 = buf554[0]
    buf556 = buf554[1]
    del buf554
    buf557 = reinterpret_tensor(buf542, (16, 512, 64), (32768, 64, 1), 0); del buf542  # reuse
    # Source Nodes: [attn_vec_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf555, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf544, (16, 512, 64), (64, 1024, 1), 0), out=buf557)
    buf558 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf559 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_82(c_void_p(buf557.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf559.data_ptr()))
    del primals_98
    buf560 = reinterpret_tensor(buf557, (1, 512, 1024), (524288, 1024, 1), 0); del buf557  # reuse
    # Source Nodes: [attn_out_39], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf558, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf559, (1, 1024, 1024), (0, 1024, 1), 0), out=buf560)
    # Source Nodes: [attn_out_40], Original ATen: [aten.native_dropout]
    buf561 = aten.native_dropout(reinterpret_tensor(buf560, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf562 = buf561[0]
    buf563 = buf561[1]
    del buf561
    buf564 = buf537; del buf537  # reuse
    buf565 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf567 = reinterpret_tensor(buf560, (512, 1, 1024), (1024, 1024, 1), 0); del buf560  # reuse
    buf568 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_83(c_void_p(buf562.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()))
    buf569 = reinterpret_tensor(buf529, (512, 4096), (4096, 1), 0); del buf529  # reuse
    # Source Nodes: [output_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_277, buf568, reinterpret_tensor(primals_276, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf569)
    del primals_277
    buf570 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_84(c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()))
    # Source Nodes: [output_107, output_108], Original ATen: [aten.gelu, aten.native_dropout]
    buf571 = aten.native_dropout(buf570, 0.1, True)
    buf572 = buf571[0]
    buf573 = buf571[1]
    del buf571
    buf574 = reinterpret_tensor(buf562, (512, 1024), (1024, 1), 0); del buf562  # reuse
    # Source Nodes: [output_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_279, reinterpret_tensor(buf572, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_278, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf574)
    del primals_279
    # Source Nodes: [output_110], Original ATen: [aten.native_dropout]
    buf575 = aten.native_dropout(reinterpret_tensor(buf574, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf576 = buf575[0]
    buf577 = buf575[1]
    del buf575
    buf578 = buf564; del buf564  # reuse
    buf579 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf581 = reinterpret_tensor(buf574, (512, 1, 1024), (1024, 1024, 1), 0); del buf574  # reuse
    buf582 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_85(c_void_p(buf576.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()))
    del primals_275
    del primals_281
    buf583 = reinterpret_tensor(buf576, (1, 512, 1024), (524288, 1024, 1), 0); del buf576  # reuse
    # Source Nodes: [q_head_h_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf582, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_99, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf583)
    buf584 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf582, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_100, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf584)
    buf585 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf582, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_101, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf585)
    buf586 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_102, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf586)
    del primals_102
    buf587 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf589 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_86(c_void_p(buf583.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf589.data_ptr()))
    del primals_103
    del primals_104
    buf588 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf587, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf584, (16, 64, 512), (64, 1, 1024), 0), out=buf588)
    buf590 = buf549; del buf549  # reuse
    # Source Nodes: [bd_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf589, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf586, (16, 64, 1024), (64, 1, 1024), 0), out=buf590)
    buf591 = buf552; del buf552  # reuse
    buf592 = reinterpret_tensor(buf588, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf588  # reuse
    buf593 = buf550; del buf550  # reuse
    buf594 = buf592; del buf592  # reuse
    cpp_fused__softmax_add_index_select_mul_87(c_void_p(buf594.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf593.data_ptr()))
    # Source Nodes: [attn_prob_28, attn_prob_29], Original ATen: [aten._softmax, aten.native_dropout]
    buf595 = aten.native_dropout(buf594, 0.1, True)
    buf596 = buf595[0]
    buf597 = buf595[1]
    del buf595
    buf598 = reinterpret_tensor(buf583, (16, 512, 64), (32768, 64, 1), 0); del buf583  # reuse
    # Source Nodes: [attn_vec_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf596, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf585, (16, 512, 64), (64, 1024, 1), 0), out=buf598)
    buf599 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf600 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_88(c_void_p(buf598.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf600.data_ptr()))
    del primals_105
    buf601 = reinterpret_tensor(buf598, (1, 512, 1024), (524288, 1024, 1), 0); del buf598  # reuse
    # Source Nodes: [attn_out_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf599, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf600, (1, 1024, 1024), (0, 1024, 1), 0), out=buf601)
    # Source Nodes: [attn_out_43], Original ATen: [aten.native_dropout]
    buf602 = aten.native_dropout(reinterpret_tensor(buf601, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf603 = buf602[0]
    buf604 = buf602[1]
    del buf602
    buf605 = buf578; del buf578  # reuse
    buf606 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf608 = reinterpret_tensor(buf601, (512, 1, 1024), (1024, 1024, 1), 0); del buf601  # reuse
    buf609 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_89(c_void_p(buf603.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()))
    buf610 = reinterpret_tensor(buf570, (512, 4096), (4096, 1), 0); del buf570  # reuse
    # Source Nodes: [output_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_285, buf609, reinterpret_tensor(primals_284, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf610)
    del primals_285
    buf611 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_90(c_void_p(buf610.data_ptr()), c_void_p(buf611.data_ptr()))
    # Source Nodes: [output_115, output_116], Original ATen: [aten.gelu, aten.native_dropout]
    buf612 = aten.native_dropout(buf611, 0.1, True)
    buf613 = buf612[0]
    buf614 = buf612[1]
    del buf612
    buf615 = reinterpret_tensor(buf603, (512, 1024), (1024, 1), 0); del buf603  # reuse
    # Source Nodes: [output_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_287, reinterpret_tensor(buf613, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_286, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf615)
    del primals_287
    # Source Nodes: [output_118], Original ATen: [aten.native_dropout]
    buf616 = aten.native_dropout(reinterpret_tensor(buf615, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf617 = buf616[0]
    buf618 = buf616[1]
    del buf616
    buf619 = buf605; del buf605  # reuse
    buf620 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf622 = reinterpret_tensor(buf615, (512, 1, 1024), (1024, 1024, 1), 0); del buf615  # reuse
    buf623 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_91(c_void_p(buf617.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf623.data_ptr()))
    del primals_283
    del primals_289
    buf624 = reinterpret_tensor(buf617, (1, 512, 1024), (524288, 1024, 1), 0); del buf617  # reuse
    # Source Nodes: [q_head_h_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf623, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_106, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf624)
    buf625 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf623, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_107, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf625)
    buf626 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf623, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_108, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf626)
    buf627 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_109, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf627)
    del primals_109
    buf628 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf630 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_92(c_void_p(buf624.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf630.data_ptr()))
    del primals_110
    del primals_111
    buf629 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf628, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf625, (16, 64, 512), (64, 1, 1024), 0), out=buf629)
    buf631 = buf590; del buf590  # reuse
    # Source Nodes: [bd_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf630, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf627, (16, 64, 1024), (64, 1, 1024), 0), out=buf631)
    buf632 = buf593; del buf593  # reuse
    buf633 = reinterpret_tensor(buf629, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf629  # reuse
    buf634 = buf591; del buf591  # reuse
    buf635 = buf633; del buf633  # reuse
    cpp_fused__softmax_add_index_select_mul_93(c_void_p(buf635.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf634.data_ptr()))
    # Source Nodes: [attn_prob_30, attn_prob_31], Original ATen: [aten._softmax, aten.native_dropout]
    buf636 = aten.native_dropout(buf635, 0.1, True)
    buf637 = buf636[0]
    buf638 = buf636[1]
    del buf636
    buf639 = reinterpret_tensor(buf624, (16, 512, 64), (32768, 64, 1), 0); del buf624  # reuse
    # Source Nodes: [attn_vec_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf637, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf626, (16, 512, 64), (64, 1024, 1), 0), out=buf639)
    buf640 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf641 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_94(c_void_p(buf639.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf641.data_ptr()))
    del primals_112
    buf642 = reinterpret_tensor(buf639, (1, 512, 1024), (524288, 1024, 1), 0); del buf639  # reuse
    # Source Nodes: [attn_out_45], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf640, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf641, (1, 1024, 1024), (0, 1024, 1), 0), out=buf642)
    # Source Nodes: [attn_out_46], Original ATen: [aten.native_dropout]
    buf643 = aten.native_dropout(reinterpret_tensor(buf642, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf644 = buf643[0]
    buf645 = buf643[1]
    del buf643
    buf646 = buf619; del buf619  # reuse
    buf647 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf649 = reinterpret_tensor(buf642, (512, 1, 1024), (1024, 1024, 1), 0); del buf642  # reuse
    buf650 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_95(c_void_p(buf644.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf650.data_ptr()))
    buf651 = reinterpret_tensor(buf611, (512, 4096), (4096, 1), 0); del buf611  # reuse
    # Source Nodes: [output_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_293, buf650, reinterpret_tensor(primals_292, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf651)
    del primals_293
    buf652 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_96(c_void_p(buf651.data_ptr()), c_void_p(buf652.data_ptr()))
    # Source Nodes: [output_123, output_124], Original ATen: [aten.gelu, aten.native_dropout]
    buf653 = aten.native_dropout(buf652, 0.1, True)
    buf654 = buf653[0]
    buf655 = buf653[1]
    del buf653
    buf656 = reinterpret_tensor(buf644, (512, 1024), (1024, 1), 0); del buf644  # reuse
    # Source Nodes: [output_125], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_295, reinterpret_tensor(buf654, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_294, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf656)
    del primals_295
    # Source Nodes: [output_126], Original ATen: [aten.native_dropout]
    buf657 = aten.native_dropout(reinterpret_tensor(buf656, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf658 = buf657[0]
    buf659 = buf657[1]
    del buf657
    buf660 = buf646; del buf646  # reuse
    buf661 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf663 = reinterpret_tensor(buf656, (512, 1, 1024), (1024, 1024, 1), 0); del buf656  # reuse
    buf664 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_97(c_void_p(buf658.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf664.data_ptr()))
    del primals_291
    del primals_297
    buf665 = reinterpret_tensor(buf658, (1, 512, 1024), (524288, 1024, 1), 0); del buf658  # reuse
    # Source Nodes: [q_head_h_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf664, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_113, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf665)
    buf666 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf664, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_114, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf666)
    buf667 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf664, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_115, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf667)
    buf668 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_116, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf668)
    del primals_116
    buf669 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf671 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_98(c_void_p(buf665.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf671.data_ptr()))
    del primals_117
    del primals_118
    buf670 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf669, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf666, (16, 64, 512), (64, 1, 1024), 0), out=buf670)
    buf672 = buf631; del buf631  # reuse
    # Source Nodes: [bd_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf671, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf668, (16, 64, 1024), (64, 1, 1024), 0), out=buf672)
    buf673 = buf634; del buf634  # reuse
    buf674 = reinterpret_tensor(buf670, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf670  # reuse
    buf675 = buf632; del buf632  # reuse
    buf676 = buf674; del buf674  # reuse
    cpp_fused__softmax_add_index_select_mul_99(c_void_p(buf676.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf675.data_ptr()))
    # Source Nodes: [attn_prob_32, attn_prob_33], Original ATen: [aten._softmax, aten.native_dropout]
    buf677 = aten.native_dropout(buf676, 0.1, True)
    buf678 = buf677[0]
    buf679 = buf677[1]
    del buf677
    buf680 = reinterpret_tensor(buf665, (16, 512, 64), (32768, 64, 1), 0); del buf665  # reuse
    # Source Nodes: [attn_vec_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf678, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf667, (16, 512, 64), (64, 1024, 1), 0), out=buf680)
    buf681 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf682 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_100(c_void_p(buf680.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()))
    del primals_119
    buf683 = reinterpret_tensor(buf680, (1, 512, 1024), (524288, 1024, 1), 0); del buf680  # reuse
    # Source Nodes: [attn_out_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf681, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf682, (1, 1024, 1024), (0, 1024, 1), 0), out=buf683)
    # Source Nodes: [attn_out_49], Original ATen: [aten.native_dropout]
    buf684 = aten.native_dropout(reinterpret_tensor(buf683, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf685 = buf684[0]
    buf686 = buf684[1]
    del buf684
    buf687 = buf660; del buf660  # reuse
    buf688 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf690 = reinterpret_tensor(buf683, (512, 1, 1024), (1024, 1024, 1), 0); del buf683  # reuse
    buf691 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_101(c_void_p(buf685.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf691.data_ptr()))
    buf692 = reinterpret_tensor(buf652, (512, 4096), (4096, 1), 0); del buf652  # reuse
    # Source Nodes: [output_130], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_301, buf691, reinterpret_tensor(primals_300, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf692)
    del primals_301
    buf693 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_102(c_void_p(buf692.data_ptr()), c_void_p(buf693.data_ptr()))
    # Source Nodes: [output_131, output_132], Original ATen: [aten.gelu, aten.native_dropout]
    buf694 = aten.native_dropout(buf693, 0.1, True)
    buf695 = buf694[0]
    buf696 = buf694[1]
    del buf694
    buf697 = reinterpret_tensor(buf685, (512, 1024), (1024, 1), 0); del buf685  # reuse
    # Source Nodes: [output_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_303, reinterpret_tensor(buf695, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_302, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf697)
    del primals_303
    # Source Nodes: [output_134], Original ATen: [aten.native_dropout]
    buf698 = aten.native_dropout(reinterpret_tensor(buf697, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf699 = buf698[0]
    buf700 = buf698[1]
    del buf698
    buf701 = buf687; del buf687  # reuse
    buf702 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf704 = reinterpret_tensor(buf697, (512, 1, 1024), (1024, 1024, 1), 0); del buf697  # reuse
    buf705 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_103(c_void_p(buf699.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(buf705.data_ptr()))
    del primals_299
    del primals_305
    buf706 = reinterpret_tensor(buf699, (1, 512, 1024), (524288, 1024, 1), 0); del buf699  # reuse
    # Source Nodes: [q_head_h_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf705, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_120, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf706)
    buf707 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf705, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_121, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf707)
    buf708 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf705, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_122, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf708)
    buf709 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_123, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf709)
    del primals_123
    buf710 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf712 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_104(c_void_p(buf706.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf712.data_ptr()))
    del primals_124
    del primals_125
    buf711 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf710, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf707, (16, 64, 512), (64, 1, 1024), 0), out=buf711)
    buf713 = buf672; del buf672  # reuse
    # Source Nodes: [bd_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf712, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf709, (16, 64, 1024), (64, 1, 1024), 0), out=buf713)
    buf714 = buf675; del buf675  # reuse
    buf715 = reinterpret_tensor(buf711, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf711  # reuse
    buf716 = buf673; del buf673  # reuse
    buf717 = buf715; del buf715  # reuse
    cpp_fused__softmax_add_index_select_mul_105(c_void_p(buf717.data_ptr()), c_void_p(buf713.data_ptr()), c_void_p(buf714.data_ptr()), c_void_p(buf716.data_ptr()))
    # Source Nodes: [attn_prob_34, attn_prob_35], Original ATen: [aten._softmax, aten.native_dropout]
    buf718 = aten.native_dropout(buf717, 0.1, True)
    buf719 = buf718[0]
    buf720 = buf718[1]
    del buf718
    buf721 = reinterpret_tensor(buf706, (16, 512, 64), (32768, 64, 1), 0); del buf706  # reuse
    # Source Nodes: [attn_vec_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf719, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf708, (16, 512, 64), (64, 1024, 1), 0), out=buf721)
    buf722 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf723 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_106(c_void_p(buf721.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(buf722.data_ptr()), c_void_p(buf723.data_ptr()))
    del primals_126
    buf724 = reinterpret_tensor(buf721, (1, 512, 1024), (524288, 1024, 1), 0); del buf721  # reuse
    # Source Nodes: [attn_out_51], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf722, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf723, (1, 1024, 1024), (0, 1024, 1), 0), out=buf724)
    # Source Nodes: [attn_out_52], Original ATen: [aten.native_dropout]
    buf725 = aten.native_dropout(reinterpret_tensor(buf724, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf726 = buf725[0]
    buf727 = buf725[1]
    del buf725
    buf728 = buf701; del buf701  # reuse
    buf729 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf731 = reinterpret_tensor(buf724, (512, 1, 1024), (1024, 1024, 1), 0); del buf724  # reuse
    buf732 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_107(c_void_p(buf726.data_ptr()), c_void_p(buf705.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(buf728.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf732.data_ptr()))
    buf733 = reinterpret_tensor(buf693, (512, 4096), (4096, 1), 0); del buf693  # reuse
    # Source Nodes: [output_138], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_309, buf732, reinterpret_tensor(primals_308, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf733)
    del primals_309
    buf734 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_108(c_void_p(buf733.data_ptr()), c_void_p(buf734.data_ptr()))
    # Source Nodes: [output_139, output_140], Original ATen: [aten.gelu, aten.native_dropout]
    buf735 = aten.native_dropout(buf734, 0.1, True)
    buf736 = buf735[0]
    buf737 = buf735[1]
    del buf735
    buf738 = reinterpret_tensor(buf726, (512, 1024), (1024, 1), 0); del buf726  # reuse
    # Source Nodes: [output_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_311, reinterpret_tensor(buf736, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_310, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf738)
    del primals_311
    # Source Nodes: [output_142], Original ATen: [aten.native_dropout]
    buf739 = aten.native_dropout(reinterpret_tensor(buf738, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf740 = buf739[0]
    buf741 = buf739[1]
    del buf739
    buf742 = buf728; del buf728  # reuse
    buf743 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf745 = reinterpret_tensor(buf738, (512, 1, 1024), (1024, 1024, 1), 0); del buf738  # reuse
    buf746 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_109(c_void_p(buf740.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(buf743.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf746.data_ptr()))
    del primals_307
    del primals_313
    buf747 = reinterpret_tensor(buf740, (1, 512, 1024), (524288, 1024, 1), 0); del buf740  # reuse
    # Source Nodes: [q_head_h_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf746, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_127, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf747)
    buf748 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf746, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_128, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf748)
    buf749 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf746, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_129, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf749)
    buf750 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_130, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf750)
    del primals_130
    buf751 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf753 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_110(c_void_p(buf747.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(buf751.data_ptr()), c_void_p(buf753.data_ptr()))
    del primals_131
    del primals_132
    buf752 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf751, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf748, (16, 64, 512), (64, 1, 1024), 0), out=buf752)
    buf754 = buf713; del buf713  # reuse
    # Source Nodes: [bd_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf753, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf750, (16, 64, 1024), (64, 1, 1024), 0), out=buf754)
    buf755 = buf716; del buf716  # reuse
    buf756 = reinterpret_tensor(buf752, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf752  # reuse
    buf757 = buf714; del buf714  # reuse
    buf758 = buf756; del buf756  # reuse
    cpp_fused__softmax_add_index_select_mul_111(c_void_p(buf758.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf757.data_ptr()))
    # Source Nodes: [attn_prob_36, attn_prob_37], Original ATen: [aten._softmax, aten.native_dropout]
    buf759 = aten.native_dropout(buf758, 0.1, True)
    buf760 = buf759[0]
    buf761 = buf759[1]
    del buf759
    buf762 = reinterpret_tensor(buf747, (16, 512, 64), (32768, 64, 1), 0); del buf747  # reuse
    # Source Nodes: [attn_vec_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf760, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf749, (16, 512, 64), (64, 1024, 1), 0), out=buf762)
    buf763 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf764 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_112(c_void_p(buf762.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(buf763.data_ptr()), c_void_p(buf764.data_ptr()))
    del primals_133
    buf765 = reinterpret_tensor(buf762, (1, 512, 1024), (524288, 1024, 1), 0); del buf762  # reuse
    # Source Nodes: [attn_out_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf763, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf764, (1, 1024, 1024), (0, 1024, 1), 0), out=buf765)
    # Source Nodes: [attn_out_55], Original ATen: [aten.native_dropout]
    buf766 = aten.native_dropout(reinterpret_tensor(buf765, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf767 = buf766[0]
    buf768 = buf766[1]
    del buf766
    buf769 = buf742; del buf742  # reuse
    buf770 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf772 = reinterpret_tensor(buf765, (512, 1, 1024), (1024, 1024, 1), 0); del buf765  # reuse
    buf773 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_113(c_void_p(buf767.data_ptr()), c_void_p(buf746.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(buf770.data_ptr()), c_void_p(buf772.data_ptr()), c_void_p(buf773.data_ptr()))
    buf774 = reinterpret_tensor(buf734, (512, 4096), (4096, 1), 0); del buf734  # reuse
    # Source Nodes: [output_146], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_317, buf773, reinterpret_tensor(primals_316, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf774)
    del primals_317
    buf775 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_114(c_void_p(buf774.data_ptr()), c_void_p(buf775.data_ptr()))
    # Source Nodes: [output_147, output_148], Original ATen: [aten.gelu, aten.native_dropout]
    buf776 = aten.native_dropout(buf775, 0.1, True)
    buf777 = buf776[0]
    buf778 = buf776[1]
    del buf776
    buf779 = reinterpret_tensor(buf767, (512, 1024), (1024, 1), 0); del buf767  # reuse
    # Source Nodes: [output_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_319, reinterpret_tensor(buf777, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_318, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf779)
    del primals_319
    # Source Nodes: [output_150], Original ATen: [aten.native_dropout]
    buf780 = aten.native_dropout(reinterpret_tensor(buf779, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf781 = buf780[0]
    buf782 = buf780[1]
    del buf780
    buf783 = buf769; del buf769  # reuse
    buf784 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf786 = reinterpret_tensor(buf779, (512, 1, 1024), (1024, 1024, 1), 0); del buf779  # reuse
    buf787 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_115(c_void_p(buf781.data_ptr()), c_void_p(buf772.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(buf783.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf787.data_ptr()))
    del primals_315
    del primals_321
    buf788 = reinterpret_tensor(buf781, (1, 512, 1024), (524288, 1024, 1), 0); del buf781  # reuse
    # Source Nodes: [q_head_h_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf787, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_134, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf788)
    buf789 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf787, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_135, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf789)
    buf790 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf787, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_136, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf790)
    buf791 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_137, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf791)
    del primals_137
    buf792 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf794 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_116(c_void_p(buf788.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(buf792.data_ptr()), c_void_p(buf794.data_ptr()))
    del primals_138
    del primals_139
    buf793 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf792, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf789, (16, 64, 512), (64, 1, 1024), 0), out=buf793)
    buf795 = buf754; del buf754  # reuse
    # Source Nodes: [bd_38], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf794, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf791, (16, 64, 1024), (64, 1, 1024), 0), out=buf795)
    buf796 = buf757; del buf757  # reuse
    buf797 = reinterpret_tensor(buf793, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf793  # reuse
    buf798 = buf755; del buf755  # reuse
    buf799 = buf797; del buf797  # reuse
    cpp_fused__softmax_add_index_select_mul_117(c_void_p(buf799.data_ptr()), c_void_p(buf795.data_ptr()), c_void_p(buf796.data_ptr()), c_void_p(buf798.data_ptr()))
    # Source Nodes: [attn_prob_38, attn_prob_39], Original ATen: [aten._softmax, aten.native_dropout]
    buf800 = aten.native_dropout(buf799, 0.1, True)
    buf801 = buf800[0]
    buf802 = buf800[1]
    del buf800
    buf803 = reinterpret_tensor(buf788, (16, 512, 64), (32768, 64, 1), 0); del buf788  # reuse
    # Source Nodes: [attn_vec_38], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf801, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf790, (16, 512, 64), (64, 1024, 1), 0), out=buf803)
    buf804 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf805 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_118(c_void_p(buf803.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf804.data_ptr()), c_void_p(buf805.data_ptr()))
    del primals_140
    buf806 = reinterpret_tensor(buf803, (1, 512, 1024), (524288, 1024, 1), 0); del buf803  # reuse
    # Source Nodes: [attn_out_57], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf804, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf805, (1, 1024, 1024), (0, 1024, 1), 0), out=buf806)
    # Source Nodes: [attn_out_58], Original ATen: [aten.native_dropout]
    buf807 = aten.native_dropout(reinterpret_tensor(buf806, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf808 = buf807[0]
    buf809 = buf807[1]
    del buf807
    buf810 = buf783; del buf783  # reuse
    buf811 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf813 = reinterpret_tensor(buf806, (512, 1, 1024), (1024, 1024, 1), 0); del buf806  # reuse
    buf814 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_119(c_void_p(buf808.data_ptr()), c_void_p(buf787.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(buf810.data_ptr()), c_void_p(buf811.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf814.data_ptr()))
    buf815 = reinterpret_tensor(buf775, (512, 4096), (4096, 1), 0); del buf775  # reuse
    # Source Nodes: [output_154], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_325, buf814, reinterpret_tensor(primals_324, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf815)
    del primals_325
    buf816 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_120(c_void_p(buf815.data_ptr()), c_void_p(buf816.data_ptr()))
    # Source Nodes: [output_155, output_156], Original ATen: [aten.gelu, aten.native_dropout]
    buf817 = aten.native_dropout(buf816, 0.1, True)
    buf818 = buf817[0]
    buf819 = buf817[1]
    del buf817
    buf820 = reinterpret_tensor(buf808, (512, 1024), (1024, 1), 0); del buf808  # reuse
    # Source Nodes: [output_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_327, reinterpret_tensor(buf818, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_326, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf820)
    del primals_327
    # Source Nodes: [output_158], Original ATen: [aten.native_dropout]
    buf821 = aten.native_dropout(reinterpret_tensor(buf820, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf822 = buf821[0]
    buf823 = buf821[1]
    del buf821
    buf824 = buf810; del buf810  # reuse
    buf825 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf827 = reinterpret_tensor(buf820, (512, 1, 1024), (1024, 1024, 1), 0); del buf820  # reuse
    buf828 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_121(c_void_p(buf822.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(buf824.data_ptr()), c_void_p(buf825.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf828.data_ptr()))
    del primals_323
    del primals_329
    buf829 = reinterpret_tensor(buf822, (1, 512, 1024), (524288, 1024, 1), 0); del buf822  # reuse
    # Source Nodes: [q_head_h_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf828, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_141, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf829)
    buf830 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf828, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_142, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf830)
    buf831 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf828, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_143, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf831)
    buf832 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_144, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf832)
    del primals_144
    buf833 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf835 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_122(c_void_p(buf829.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf833.data_ptr()), c_void_p(buf835.data_ptr()))
    del primals_145
    del primals_146
    buf834 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf833, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf830, (16, 64, 512), (64, 1, 1024), 0), out=buf834)
    buf836 = buf795; del buf795  # reuse
    # Source Nodes: [bd_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf835, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf832, (16, 64, 1024), (64, 1, 1024), 0), out=buf836)
    buf837 = buf798; del buf798  # reuse
    buf838 = reinterpret_tensor(buf834, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf834  # reuse
    buf839 = buf796; del buf796  # reuse
    buf840 = buf838; del buf838  # reuse
    cpp_fused__softmax_add_index_select_mul_123(c_void_p(buf840.data_ptr()), c_void_p(buf836.data_ptr()), c_void_p(buf837.data_ptr()), c_void_p(buf839.data_ptr()))
    # Source Nodes: [attn_prob_40, attn_prob_41], Original ATen: [aten._softmax, aten.native_dropout]
    buf841 = aten.native_dropout(buf840, 0.1, True)
    buf842 = buf841[0]
    buf843 = buf841[1]
    del buf841
    buf844 = reinterpret_tensor(buf829, (16, 512, 64), (32768, 64, 1), 0); del buf829  # reuse
    # Source Nodes: [attn_vec_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf842, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf831, (16, 512, 64), (64, 1024, 1), 0), out=buf844)
    buf845 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf846 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_124(c_void_p(buf844.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf845.data_ptr()), c_void_p(buf846.data_ptr()))
    del primals_147
    buf847 = reinterpret_tensor(buf844, (1, 512, 1024), (524288, 1024, 1), 0); del buf844  # reuse
    # Source Nodes: [attn_out_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf845, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf846, (1, 1024, 1024), (0, 1024, 1), 0), out=buf847)
    # Source Nodes: [attn_out_61], Original ATen: [aten.native_dropout]
    buf848 = aten.native_dropout(reinterpret_tensor(buf847, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf849 = buf848[0]
    buf850 = buf848[1]
    del buf848
    buf851 = buf824; del buf824  # reuse
    buf852 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf854 = reinterpret_tensor(buf847, (512, 1, 1024), (1024, 1024, 1), 0); del buf847  # reuse
    buf855 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_125(c_void_p(buf849.data_ptr()), c_void_p(buf828.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(buf851.data_ptr()), c_void_p(buf852.data_ptr()), c_void_p(buf854.data_ptr()), c_void_p(buf855.data_ptr()))
    buf856 = reinterpret_tensor(buf816, (512, 4096), (4096, 1), 0); del buf816  # reuse
    # Source Nodes: [output_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_333, buf855, reinterpret_tensor(primals_332, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf856)
    del primals_333
    buf857 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_126(c_void_p(buf856.data_ptr()), c_void_p(buf857.data_ptr()))
    # Source Nodes: [output_163, output_164], Original ATen: [aten.gelu, aten.native_dropout]
    buf858 = aten.native_dropout(buf857, 0.1, True)
    buf859 = buf858[0]
    buf860 = buf858[1]
    del buf858
    buf861 = reinterpret_tensor(buf849, (512, 1024), (1024, 1), 0); del buf849  # reuse
    # Source Nodes: [output_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_335, reinterpret_tensor(buf859, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_334, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf861)
    del primals_335
    # Source Nodes: [output_166], Original ATen: [aten.native_dropout]
    buf862 = aten.native_dropout(reinterpret_tensor(buf861, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf863 = buf862[0]
    buf864 = buf862[1]
    del buf862
    buf865 = buf851; del buf851  # reuse
    buf866 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf868 = reinterpret_tensor(buf861, (512, 1, 1024), (1024, 1024, 1), 0); del buf861  # reuse
    buf869 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_127(c_void_p(buf863.data_ptr()), c_void_p(buf854.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(buf865.data_ptr()), c_void_p(buf866.data_ptr()), c_void_p(buf868.data_ptr()), c_void_p(buf869.data_ptr()))
    del primals_331
    del primals_337
    buf870 = reinterpret_tensor(buf863, (1, 512, 1024), (524288, 1024, 1), 0); del buf863  # reuse
    # Source Nodes: [q_head_h_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf869, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_148, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf870)
    buf871 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf869, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_149, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf871)
    buf872 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf869, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_150, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf872)
    buf873 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_151, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf873)
    del primals_151
    buf874 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf876 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_128(c_void_p(buf870.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf874.data_ptr()), c_void_p(buf876.data_ptr()))
    del primals_152
    del primals_153
    buf875 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf874, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf871, (16, 64, 512), (64, 1, 1024), 0), out=buf875)
    buf877 = buf836; del buf836  # reuse
    # Source Nodes: [bd_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf876, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf873, (16, 64, 1024), (64, 1, 1024), 0), out=buf877)
    buf878 = buf839; del buf839  # reuse
    buf879 = reinterpret_tensor(buf875, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf875  # reuse
    buf880 = buf837; del buf837  # reuse
    buf881 = buf879; del buf879  # reuse
    cpp_fused__softmax_add_index_select_mul_129(c_void_p(buf881.data_ptr()), c_void_p(buf877.data_ptr()), c_void_p(buf878.data_ptr()), c_void_p(buf880.data_ptr()))
    # Source Nodes: [attn_prob_42, attn_prob_43], Original ATen: [aten._softmax, aten.native_dropout]
    buf882 = aten.native_dropout(buf881, 0.1, True)
    buf883 = buf882[0]
    buf884 = buf882[1]
    del buf882
    buf885 = reinterpret_tensor(buf870, (16, 512, 64), (32768, 64, 1), 0); del buf870  # reuse
    # Source Nodes: [attn_vec_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf883, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf872, (16, 512, 64), (64, 1024, 1), 0), out=buf885)
    buf886 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf887 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_130(c_void_p(buf885.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(buf886.data_ptr()), c_void_p(buf887.data_ptr()))
    del primals_154
    buf888 = reinterpret_tensor(buf885, (1, 512, 1024), (524288, 1024, 1), 0); del buf885  # reuse
    # Source Nodes: [attn_out_63], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf886, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf887, (1, 1024, 1024), (0, 1024, 1), 0), out=buf888)
    # Source Nodes: [attn_out_64], Original ATen: [aten.native_dropout]
    buf889 = aten.native_dropout(reinterpret_tensor(buf888, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf890 = buf889[0]
    buf891 = buf889[1]
    del buf889
    buf892 = buf865; del buf865  # reuse
    buf893 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf895 = reinterpret_tensor(buf888, (512, 1, 1024), (1024, 1024, 1), 0); del buf888  # reuse
    buf896 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_131(c_void_p(buf890.data_ptr()), c_void_p(buf869.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(buf892.data_ptr()), c_void_p(buf893.data_ptr()), c_void_p(buf895.data_ptr()), c_void_p(buf896.data_ptr()))
    buf897 = reinterpret_tensor(buf857, (512, 4096), (4096, 1), 0); del buf857  # reuse
    # Source Nodes: [output_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_341, buf896, reinterpret_tensor(primals_340, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf897)
    del primals_341
    buf898 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_132(c_void_p(buf897.data_ptr()), c_void_p(buf898.data_ptr()))
    # Source Nodes: [output_171, output_172], Original ATen: [aten.gelu, aten.native_dropout]
    buf899 = aten.native_dropout(buf898, 0.1, True)
    buf900 = buf899[0]
    buf901 = buf899[1]
    del buf899
    buf902 = reinterpret_tensor(buf890, (512, 1024), (1024, 1), 0); del buf890  # reuse
    # Source Nodes: [output_173], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_343, reinterpret_tensor(buf900, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_342, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf902)
    del primals_343
    # Source Nodes: [output_174], Original ATen: [aten.native_dropout]
    buf903 = aten.native_dropout(reinterpret_tensor(buf902, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf904 = buf903[0]
    buf905 = buf903[1]
    del buf903
    buf906 = buf892; del buf892  # reuse
    buf907 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf909 = reinterpret_tensor(buf902, (512, 1, 1024), (1024, 1024, 1), 0); del buf902  # reuse
    buf910 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_133(c_void_p(buf904.data_ptr()), c_void_p(buf895.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(buf906.data_ptr()), c_void_p(buf907.data_ptr()), c_void_p(buf909.data_ptr()), c_void_p(buf910.data_ptr()))
    del primals_339
    del primals_345
    buf911 = reinterpret_tensor(buf904, (1, 512, 1024), (524288, 1024, 1), 0); del buf904  # reuse
    # Source Nodes: [q_head_h_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf910, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_155, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf911)
    buf912 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf910, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_156, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf912)
    buf913 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf910, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_157, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf913)
    buf914 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_158, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf914)
    del primals_158
    buf915 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf917 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_134(c_void_p(buf911.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(buf915.data_ptr()), c_void_p(buf917.data_ptr()))
    del primals_159
    del primals_160
    buf916 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf915, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf912, (16, 64, 512), (64, 1, 1024), 0), out=buf916)
    buf918 = buf877; del buf877  # reuse
    # Source Nodes: [bd_44], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf917, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf914, (16, 64, 1024), (64, 1, 1024), 0), out=buf918)
    buf919 = buf880; del buf880  # reuse
    buf920 = reinterpret_tensor(buf916, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf916  # reuse
    buf921 = buf878; del buf878  # reuse
    buf922 = buf920; del buf920  # reuse
    cpp_fused__softmax_add_index_select_mul_135(c_void_p(buf922.data_ptr()), c_void_p(buf918.data_ptr()), c_void_p(buf919.data_ptr()), c_void_p(buf921.data_ptr()))
    # Source Nodes: [attn_prob_44, attn_prob_45], Original ATen: [aten._softmax, aten.native_dropout]
    buf923 = aten.native_dropout(buf922, 0.1, True)
    buf924 = buf923[0]
    buf925 = buf923[1]
    del buf923
    buf926 = reinterpret_tensor(buf911, (16, 512, 64), (32768, 64, 1), 0); del buf911  # reuse
    # Source Nodes: [attn_vec_44], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf924, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf913, (16, 512, 64), (64, 1024, 1), 0), out=buf926)
    buf927 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf928 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_136(c_void_p(buf926.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(buf927.data_ptr()), c_void_p(buf928.data_ptr()))
    del primals_161
    buf929 = reinterpret_tensor(buf926, (1, 512, 1024), (524288, 1024, 1), 0); del buf926  # reuse
    # Source Nodes: [attn_out_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf927, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf928, (1, 1024, 1024), (0, 1024, 1), 0), out=buf929)
    # Source Nodes: [attn_out_67], Original ATen: [aten.native_dropout]
    buf930 = aten.native_dropout(reinterpret_tensor(buf929, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf931 = buf930[0]
    buf932 = buf930[1]
    del buf930
    buf933 = buf906; del buf906  # reuse
    buf934 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf936 = reinterpret_tensor(buf929, (512, 1, 1024), (1024, 1024, 1), 0); del buf929  # reuse
    buf937 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_137(c_void_p(buf931.data_ptr()), c_void_p(buf910.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(buf933.data_ptr()), c_void_p(buf934.data_ptr()), c_void_p(buf936.data_ptr()), c_void_p(buf937.data_ptr()))
    buf938 = reinterpret_tensor(buf898, (512, 4096), (4096, 1), 0); del buf898  # reuse
    # Source Nodes: [output_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_349, buf937, reinterpret_tensor(primals_348, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf938)
    del primals_349
    buf939 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_138(c_void_p(buf938.data_ptr()), c_void_p(buf939.data_ptr()))
    # Source Nodes: [output_179, output_180], Original ATen: [aten.gelu, aten.native_dropout]
    buf940 = aten.native_dropout(buf939, 0.1, True)
    buf941 = buf940[0]
    buf942 = buf940[1]
    del buf940
    buf943 = reinterpret_tensor(buf931, (512, 1024), (1024, 1), 0); del buf931  # reuse
    # Source Nodes: [output_181], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_351, reinterpret_tensor(buf941, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_350, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf943)
    del primals_351
    # Source Nodes: [output_182], Original ATen: [aten.native_dropout]
    buf944 = aten.native_dropout(reinterpret_tensor(buf943, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf945 = buf944[0]
    buf946 = buf944[1]
    del buf944
    buf947 = buf933; del buf933  # reuse
    buf948 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf950 = reinterpret_tensor(buf943, (512, 1, 1024), (1024, 1024, 1), 0); del buf943  # reuse
    buf951 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_139(c_void_p(buf945.data_ptr()), c_void_p(buf936.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(buf947.data_ptr()), c_void_p(buf948.data_ptr()), c_void_p(buf950.data_ptr()), c_void_p(buf951.data_ptr()))
    del primals_347
    del primals_353
    buf952 = reinterpret_tensor(buf945, (1, 512, 1024), (524288, 1024, 1), 0); del buf945  # reuse
    # Source Nodes: [q_head_h_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf951, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_162, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf952)
    buf953 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf951, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_163, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf953)
    buf954 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_head_h_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf951, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_164, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf954)
    buf955 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_165, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf955)
    del primals_165
    buf956 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf958 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_140(c_void_p(buf952.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf956.data_ptr()), c_void_p(buf958.data_ptr()))
    del primals_166
    del primals_167
    buf957 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf956, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf953, (16, 64, 512), (64, 1, 1024), 0), out=buf957)
    buf959 = buf918; del buf918  # reuse
    # Source Nodes: [bd_46], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf958, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf955, (16, 64, 1024), (64, 1, 1024), 0), out=buf959)
    buf960 = buf921; del buf921  # reuse
    buf961 = reinterpret_tensor(buf957, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf957  # reuse
    buf962 = buf919; del buf919  # reuse
    buf963 = buf961; del buf961  # reuse
    cpp_fused__softmax_add_index_select_mul_141(c_void_p(buf963.data_ptr()), c_void_p(buf959.data_ptr()), c_void_p(buf960.data_ptr()), c_void_p(buf962.data_ptr()))
    del buf959
    del buf960
    del buf962
    # Source Nodes: [attn_prob_46, attn_prob_47], Original ATen: [aten._softmax, aten.native_dropout]
    buf964 = aten.native_dropout(buf963, 0.1, True)
    buf965 = buf964[0]
    buf966 = buf964[1]
    del buf964
    buf967 = reinterpret_tensor(buf952, (16, 512, 64), (32768, 64, 1), 0); del buf952  # reuse
    # Source Nodes: [attn_vec_46], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf965, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf954, (16, 512, 64), (64, 1024, 1), 0), out=buf967)
    buf968 = empty((512, 64, 16, 1, 1), device='cpu', dtype=torch.float32)
    buf969 = empty((64, 16, 1, 1024, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_142(c_void_p(buf967.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(buf968.data_ptr()), c_void_p(buf969.data_ptr()))
    del primals_168
    buf970 = reinterpret_tensor(buf967, (1, 512, 1024), (524288, 1024, 1), 0); del buf967  # reuse
    # Source Nodes: [attn_out_69], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf968, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf969, (1, 1024, 1024), (0, 1024, 1), 0), out=buf970)
    # Source Nodes: [attn_out_70], Original ATen: [aten.native_dropout]
    buf971 = aten.native_dropout(reinterpret_tensor(buf970, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf972 = buf971[0]
    buf973 = buf971[1]
    del buf971
    buf974 = buf947; del buf947  # reuse
    buf975 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf977 = reinterpret_tensor(buf970, (512, 1, 1024), (1024, 1024, 1), 0); del buf970  # reuse
    buf978 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_143(c_void_p(buf972.data_ptr()), c_void_p(buf951.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(buf974.data_ptr()), c_void_p(buf975.data_ptr()), c_void_p(buf977.data_ptr()), c_void_p(buf978.data_ptr()))
    buf979 = reinterpret_tensor(buf939, (512, 4096), (4096, 1), 0); del buf939  # reuse
    # Source Nodes: [output_186], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_357, buf978, reinterpret_tensor(primals_356, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf979)
    del primals_357
    buf980 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_144(c_void_p(buf979.data_ptr()), c_void_p(buf980.data_ptr()))
    # Source Nodes: [output_187, output_188], Original ATen: [aten.gelu, aten.native_dropout]
    buf981 = aten.native_dropout(buf980, 0.1, True)
    del buf980
    buf982 = buf981[0]
    buf983 = buf981[1]
    del buf981
    buf984 = reinterpret_tensor(buf972, (512, 1024), (1024, 1), 0); del buf972  # reuse
    # Source Nodes: [output_189], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_359, reinterpret_tensor(buf982, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_358, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf984)
    del primals_359
    # Source Nodes: [output_190], Original ATen: [aten.native_dropout]
    buf985 = aten.native_dropout(reinterpret_tensor(buf984, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
    buf986 = buf985[0]
    buf987 = buf985[1]
    del buf985
    buf988 = buf974; del buf974  # reuse
    buf989 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf991 = reinterpret_tensor(buf984, (512, 1, 1024), (1024, 1024, 1), 0); del buf984  # reuse
    buf992 = empty_strided((512, 1, 1024), (1024, 524288, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_145(c_void_p(buf986.data_ptr()), c_void_p(buf977.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(buf988.data_ptr()), c_void_p(buf989.data_ptr()), c_void_p(buf991.data_ptr()), c_void_p(buf992.data_ptr()))
    del buf986
    del primals_355
    del primals_361
    # Source Nodes: [output_192, output_h_96], Original ATen: [aten.native_dropout, aten.native_layer_norm]
    buf993 = aten.native_dropout(buf992, 0.1, True)
    del buf992
    buf994 = buf993[0]
    buf995 = buf993[1]
    del buf993
    buf996 = empty((512, 32000), device='cpu', dtype=torch.float32)
    # Source Nodes: [logits], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_363, reinterpret_tensor(buf994, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_362, (1024, 32000), (1, 1024), 0), alpha=1, beta=1, out=buf996)
    del primals_363
    buf997 = reinterpret_tensor(buf988, (512, 1), (1, 512), 0); del buf988  # reuse
    buf998 = empty_strided((512, 1), (1, 512), device='cpu', dtype=torch.float32)
    buf999 = empty((512, 32000), device='cpu', dtype=torch.float32)
    buf1000 = empty((), device='cpu', dtype=torch.int64)
    buf1002 = empty((), device='cpu', dtype=torch.float32)
    buf1001 = empty((), device='cpu', dtype=torch.float32)
    buf1051 = buf1002; del buf1002  # reuse
    buf1003 = reinterpret_tensor(buf989, (512, 1, 1), (1, 1, 1), 0); del buf989  # reuse
    buf1004 = reinterpret_tensor(buf975, (512, 1, 1), (1, 1, 1), 0); del buf975  # reuse
    buf1005 = reinterpret_tensor(buf948, (512, 1, 1), (1, 1, 1), 0); del buf948  # reuse
    buf1006 = reinterpret_tensor(buf934, (512, 1, 1), (1, 1, 1), 0); del buf934  # reuse
    buf1007 = reinterpret_tensor(buf907, (512, 1, 1), (1, 1, 1), 0); del buf907  # reuse
    buf1008 = reinterpret_tensor(buf893, (512, 1, 1), (1, 1, 1), 0); del buf893  # reuse
    buf1009 = reinterpret_tensor(buf866, (512, 1, 1), (1, 1, 1), 0); del buf866  # reuse
    buf1010 = reinterpret_tensor(buf852, (512, 1, 1), (1, 1, 1), 0); del buf852  # reuse
    buf1011 = reinterpret_tensor(buf825, (512, 1, 1), (1, 1, 1), 0); del buf825  # reuse
    buf1012 = reinterpret_tensor(buf811, (512, 1, 1), (1, 1, 1), 0); del buf811  # reuse
    buf1013 = reinterpret_tensor(buf784, (512, 1, 1), (1, 1, 1), 0); del buf784  # reuse
    buf1014 = reinterpret_tensor(buf770, (512, 1, 1), (1, 1, 1), 0); del buf770  # reuse
    buf1015 = reinterpret_tensor(buf743, (512, 1, 1), (1, 1, 1), 0); del buf743  # reuse
    buf1016 = reinterpret_tensor(buf729, (512, 1, 1), (1, 1, 1), 0); del buf729  # reuse
    buf1017 = reinterpret_tensor(buf702, (512, 1, 1), (1, 1, 1), 0); del buf702  # reuse
    buf1018 = reinterpret_tensor(buf688, (512, 1, 1), (1, 1, 1), 0); del buf688  # reuse
    buf1019 = reinterpret_tensor(buf661, (512, 1, 1), (1, 1, 1), 0); del buf661  # reuse
    buf1020 = reinterpret_tensor(buf647, (512, 1, 1), (1, 1, 1), 0); del buf647  # reuse
    buf1021 = reinterpret_tensor(buf620, (512, 1, 1), (1, 1, 1), 0); del buf620  # reuse
    buf1022 = reinterpret_tensor(buf606, (512, 1, 1), (1, 1, 1), 0); del buf606  # reuse
    buf1023 = reinterpret_tensor(buf579, (512, 1, 1), (1, 1, 1), 0); del buf579  # reuse
    buf1024 = reinterpret_tensor(buf565, (512, 1, 1), (1, 1, 1), 0); del buf565  # reuse
    buf1025 = reinterpret_tensor(buf538, (512, 1, 1), (1, 1, 1), 0); del buf538  # reuse
    buf1026 = reinterpret_tensor(buf524, (512, 1, 1), (1, 1, 1), 0); del buf524  # reuse
    buf1027 = reinterpret_tensor(buf497, (512, 1, 1), (1, 1, 1), 0); del buf497  # reuse
    buf1028 = reinterpret_tensor(buf483, (512, 1, 1), (1, 1, 1), 0); del buf483  # reuse
    buf1029 = reinterpret_tensor(buf456, (512, 1, 1), (1, 1, 1), 0); del buf456  # reuse
    buf1030 = reinterpret_tensor(buf442, (512, 1, 1), (1, 1, 1), 0); del buf442  # reuse
    buf1031 = reinterpret_tensor(buf415, (512, 1, 1), (1, 1, 1), 0); del buf415  # reuse
    buf1032 = reinterpret_tensor(buf401, (512, 1, 1), (1, 1, 1), 0); del buf401  # reuse
    buf1033 = reinterpret_tensor(buf374, (512, 1, 1), (1, 1, 1), 0); del buf374  # reuse
    buf1034 = reinterpret_tensor(buf360, (512, 1, 1), (1, 1, 1), 0); del buf360  # reuse
    buf1035 = reinterpret_tensor(buf333, (512, 1, 1), (1, 1, 1), 0); del buf333  # reuse
    buf1036 = reinterpret_tensor(buf319, (512, 1, 1), (1, 1, 1), 0); del buf319  # reuse
    buf1037 = reinterpret_tensor(buf292, (512, 1, 1), (1, 1, 1), 0); del buf292  # reuse
    buf1038 = reinterpret_tensor(buf278, (512, 1, 1), (1, 1, 1), 0); del buf278  # reuse
    buf1039 = reinterpret_tensor(buf251, (512, 1, 1), (1, 1, 1), 0); del buf251  # reuse
    buf1040 = reinterpret_tensor(buf237, (512, 1, 1), (1, 1, 1), 0); del buf237  # reuse
    buf1041 = reinterpret_tensor(buf210, (512, 1, 1), (1, 1, 1), 0); del buf210  # reuse
    buf1042 = reinterpret_tensor(buf196, (512, 1, 1), (1, 1, 1), 0); del buf196  # reuse
    buf1043 = reinterpret_tensor(buf169, (512, 1, 1), (1, 1, 1), 0); del buf169  # reuse
    buf1044 = reinterpret_tensor(buf155, (512, 1, 1), (1, 1, 1), 0); del buf155  # reuse
    buf1045 = reinterpret_tensor(buf128, (512, 1, 1), (1, 1, 1), 0); del buf128  # reuse
    buf1046 = reinterpret_tensor(buf114, (512, 1, 1), (1, 1, 1), 0); del buf114  # reuse
    buf1047 = reinterpret_tensor(buf87, (512, 1, 1), (1, 1, 1), 0); del buf87  # reuse
    buf1048 = reinterpret_tensor(buf73, (512, 1, 1), (1, 1, 1), 0); del buf73  # reuse
    buf1049 = reinterpret_tensor(buf46, (512, 1, 1), (1, 1, 1), 0); del buf46  # reuse
    buf1050 = reinterpret_tensor(buf32, (512, 1, 1), (1, 1, 1), 0); del buf32  # reuse
    cpp_fused__log_softmax_add_native_layer_norm_native_layer_norm_backward_nll_loss_forward_146(c_void_p(buf1051.data_ptr()), c_void_p(buf1003.data_ptr()), c_void_p(buf1004.data_ptr()), c_void_p(buf1005.data_ptr()), c_void_p(buf1006.data_ptr()), c_void_p(buf1007.data_ptr()), c_void_p(buf1008.data_ptr()), c_void_p(buf1009.data_ptr()), c_void_p(buf1010.data_ptr()), c_void_p(buf1011.data_ptr()), c_void_p(buf1012.data_ptr()), c_void_p(buf1013.data_ptr()), c_void_p(buf1014.data_ptr()), c_void_p(buf1015.data_ptr()), c_void_p(buf1016.data_ptr()), c_void_p(buf1017.data_ptr()), c_void_p(buf1018.data_ptr()), c_void_p(buf1019.data_ptr()), c_void_p(buf1020.data_ptr()), c_void_p(buf1021.data_ptr()), c_void_p(buf1022.data_ptr()), c_void_p(buf1023.data_ptr()), c_void_p(buf1024.data_ptr()), c_void_p(buf1025.data_ptr()), c_void_p(buf1026.data_ptr()), c_void_p(buf1027.data_ptr()), c_void_p(buf1028.data_ptr()), c_void_p(buf1029.data_ptr()), c_void_p(buf1030.data_ptr()), c_void_p(buf1031.data_ptr()), c_void_p(buf1032.data_ptr()), c_void_p(buf1033.data_ptr()), c_void_p(buf1034.data_ptr()), c_void_p(buf1035.data_ptr()), c_void_p(buf1036.data_ptr()), c_void_p(buf1037.data_ptr()), c_void_p(buf1038.data_ptr()), c_void_p(buf1039.data_ptr()), c_void_p(buf1040.data_ptr()), c_void_p(buf1041.data_ptr()), c_void_p(buf1042.data_ptr()), c_void_p(buf1043.data_ptr()), c_void_p(buf1044.data_ptr()), c_void_p(buf1045.data_ptr()), c_void_p(buf1046.data_ptr()), c_void_p(buf1047.data_ptr()), c_void_p(buf1048.data_ptr()), c_void_p(buf1049.data_ptr()), c_void_p(buf1050.data_ptr()), c_void_p(buf996.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(buf997.data_ptr()), c_void_p(buf998.data_ptr()), c_void_p(buf999.data_ptr()), c_void_p(buf1000.data_ptr()), c_void_p(buf1001.data_ptr()))
    return (buf1051, reinterpret_tensor(buf996, (1, 512, 32000), (16384000, 32000, 1), 0), primals_170, primals_176, primals_178, primals_184, primals_186, primals_192, primals_194, primals_200, primals_202, primals_208, primals_210, primals_216, primals_218, primals_224, primals_226, primals_232, primals_234, primals_240, primals_242, primals_248, primals_250, primals_256, primals_258, primals_264, primals_266, primals_272, primals_274, primals_280, primals_282, primals_288, primals_290, primals_296, primals_298, primals_304, primals_306, primals_312, primals_314, primals_320, primals_322, primals_328, primals_330, primals_336, primals_338, primals_344, primals_346, primals_352, primals_354, primals_360, primals_365, reinterpret_tensor(primals_364, (512, 1), (1, 512), 0), buf3, buf4, buf23, buf30, buf34, buf35, buf36, buf40, reinterpret_tensor(buf39, (512, 4096), (4096, 1), 0), buf44, buf48, buf64, buf71, buf75, buf76, buf77, buf81, reinterpret_tensor(buf80, (512, 4096), (4096, 1), 0), buf85, buf89, buf105, buf112, buf116, buf117, buf118, buf122, reinterpret_tensor(buf121, (512, 4096), (4096, 1), 0), buf126, buf130, buf146, buf153, buf157, buf158, buf159, buf163, reinterpret_tensor(buf162, (512, 4096), (4096, 1), 0), buf167, buf171, buf187, buf194, buf198, buf199, buf200, buf204, reinterpret_tensor(buf203, (512, 4096), (4096, 1), 0), buf208, buf212, buf228, buf235, buf239, buf240, buf241, buf245, reinterpret_tensor(buf244, (512, 4096), (4096, 1), 0), buf249, buf253, buf269, buf276, buf280, buf281, buf282, buf286, reinterpret_tensor(buf285, (512, 4096), (4096, 1), 0), buf290, buf294, buf310, buf317, buf321, buf322, buf323, buf327, reinterpret_tensor(buf326, (512, 4096), (4096, 1), 0), buf331, buf335, buf351, buf358, buf362, buf363, buf364, buf368, reinterpret_tensor(buf367, (512, 4096), (4096, 1), 0), buf372, buf376, buf392, buf399, buf403, buf404, buf405, buf409, reinterpret_tensor(buf408, (512, 4096), (4096, 1), 0), buf413, buf417, buf433, buf440, buf444, buf445, buf446, buf450, reinterpret_tensor(buf449, (512, 4096), (4096, 1), 0), buf454, buf458, buf474, buf481, buf485, buf486, buf487, buf491, reinterpret_tensor(buf490, (512, 4096), (4096, 1), 0), buf495, buf499, buf515, buf522, buf526, buf527, buf528, buf532, reinterpret_tensor(buf531, (512, 4096), (4096, 1), 0), buf536, buf540, buf556, buf563, buf567, buf568, buf569, buf573, reinterpret_tensor(buf572, (512, 4096), (4096, 1), 0), buf577, buf581, buf597, buf604, buf608, buf609, buf610, buf614, reinterpret_tensor(buf613, (512, 4096), (4096, 1), 0), buf618, buf622, buf638, buf645, buf649, buf650, buf651, buf655, reinterpret_tensor(buf654, (512, 4096), (4096, 1), 0), buf659, buf663, buf679, buf686, buf690, buf691, buf692, buf696, reinterpret_tensor(buf695, (512, 4096), (4096, 1), 0), buf700, buf704, buf720, buf727, buf731, buf732, buf733, buf737, reinterpret_tensor(buf736, (512, 4096), (4096, 1), 0), buf741, buf745, buf761, buf768, buf772, buf773, buf774, buf778, reinterpret_tensor(buf777, (512, 4096), (4096, 1), 0), buf782, buf786, buf802, buf809, buf813, buf814, buf815, buf819, reinterpret_tensor(buf818, (512, 4096), (4096, 1), 0), buf823, buf827, buf843, buf850, buf854, buf855, buf856, buf860, reinterpret_tensor(buf859, (512, 4096), (4096, 1), 0), buf864, buf868, buf884, buf891, buf895, buf896, buf897, buf901, reinterpret_tensor(buf900, (512, 4096), (4096, 1), 0), buf905, buf909, buf925, buf932, buf936, buf937, buf938, buf942, reinterpret_tensor(buf941, (512, 4096), (4096, 1), 0), buf946, buf950, buf966, buf973, buf977, buf978, buf979, buf983, reinterpret_tensor(buf982, (512, 4096), (4096, 1), 0), buf987, buf991, buf995, reinterpret_tensor(buf994, (512, 1024), (1024, 1), 0), buf999, buf1001, reinterpret_tensor(primals_362, (32000, 1024), (1024, 1), 0), buf1003, reinterpret_tensor(primals_358, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_356, (4096, 1024), (1024, 1), 0), buf1004, reinterpret_tensor(buf968, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf969, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf965, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf954, (16, 64, 512), (64, 1, 1024), 0), buf963, reinterpret_tensor(buf958, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf955, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf956, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf953, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf951, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_164, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_163, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_162, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1005, reinterpret_tensor(primals_350, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_348, (4096, 1024), (1024, 1), 0), buf1006, reinterpret_tensor(buf927, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf928, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf924, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf913, (16, 64, 512), (64, 1, 1024), 0), buf922, reinterpret_tensor(buf917, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf914, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf915, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf912, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf910, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_157, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_156, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_155, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1007, reinterpret_tensor(primals_342, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_340, (4096, 1024), (1024, 1), 0), buf1008, reinterpret_tensor(buf886, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf887, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf883, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf872, (16, 64, 512), (64, 1, 1024), 0), buf881, reinterpret_tensor(buf876, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf873, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf874, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf871, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf869, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_150, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_149, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_148, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1009, reinterpret_tensor(primals_334, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_332, (4096, 1024), (1024, 1), 0), buf1010, reinterpret_tensor(buf845, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf846, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf842, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf831, (16, 64, 512), (64, 1, 1024), 0), buf840, reinterpret_tensor(buf835, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf832, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf833, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf830, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf828, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_143, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_142, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_141, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1011, reinterpret_tensor(primals_326, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_324, (4096, 1024), (1024, 1), 0), buf1012, reinterpret_tensor(buf804, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf805, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf801, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf790, (16, 64, 512), (64, 1, 1024), 0), buf799, reinterpret_tensor(buf794, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf791, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf792, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf789, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf787, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_136, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_135, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_134, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1013, reinterpret_tensor(primals_318, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_316, (4096, 1024), (1024, 1), 0), buf1014, reinterpret_tensor(buf763, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf764, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf760, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf749, (16, 64, 512), (64, 1, 1024), 0), buf758, reinterpret_tensor(buf753, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf750, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf751, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf748, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf746, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_129, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_128, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_127, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1015, reinterpret_tensor(primals_310, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_308, (4096, 1024), (1024, 1), 0), buf1016, reinterpret_tensor(buf722, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf723, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf719, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf708, (16, 64, 512), (64, 1, 1024), 0), buf717, reinterpret_tensor(buf712, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf709, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf710, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf707, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf705, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_122, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_121, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_120, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1017, reinterpret_tensor(primals_302, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_300, (4096, 1024), (1024, 1), 0), buf1018, reinterpret_tensor(buf681, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf682, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf678, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf667, (16, 64, 512), (64, 1, 1024), 0), buf676, reinterpret_tensor(buf671, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf668, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf669, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf666, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf664, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_115, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_114, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_113, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1019, reinterpret_tensor(primals_294, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_292, (4096, 1024), (1024, 1), 0), buf1020, reinterpret_tensor(buf640, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf641, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf637, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf626, (16, 64, 512), (64, 1, 1024), 0), buf635, reinterpret_tensor(buf630, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf627, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf628, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf625, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf623, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_108, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_107, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_106, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1021, reinterpret_tensor(primals_286, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_284, (4096, 1024), (1024, 1), 0), buf1022, reinterpret_tensor(buf599, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf600, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf596, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf585, (16, 64, 512), (64, 1, 1024), 0), buf594, reinterpret_tensor(buf589, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf586, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf587, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf584, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf582, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_101, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_100, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_99, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1023, reinterpret_tensor(primals_278, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_276, (4096, 1024), (1024, 1), 0), buf1024, reinterpret_tensor(buf558, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf559, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf555, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf544, (16, 64, 512), (64, 1, 1024), 0), buf553, reinterpret_tensor(buf548, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf545, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf546, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf543, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf541, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_94, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_93, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_92, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1025, reinterpret_tensor(primals_270, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_268, (4096, 1024), (1024, 1), 0), buf1026, reinterpret_tensor(buf517, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf518, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf514, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf503, (16, 64, 512), (64, 1, 1024), 0), buf512, reinterpret_tensor(buf507, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf504, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf505, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf502, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf500, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_87, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_86, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_85, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1027, reinterpret_tensor(primals_262, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_260, (4096, 1024), (1024, 1), 0), buf1028, reinterpret_tensor(buf476, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf477, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf473, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf462, (16, 64, 512), (64, 1, 1024), 0), buf471, reinterpret_tensor(buf466, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf463, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf464, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf461, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf459, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_80, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_79, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_78, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1029, reinterpret_tensor(primals_254, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_252, (4096, 1024), (1024, 1), 0), buf1030, reinterpret_tensor(buf435, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf436, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf432, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf421, (16, 64, 512), (64, 1, 1024), 0), buf430, reinterpret_tensor(buf425, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf422, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf423, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf420, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf418, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_73, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_72, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_71, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1031, reinterpret_tensor(primals_246, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_244, (4096, 1024), (1024, 1), 0), buf1032, reinterpret_tensor(buf394, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf395, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf391, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf380, (16, 64, 512), (64, 1, 1024), 0), buf389, reinterpret_tensor(buf384, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf381, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf382, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf379, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf377, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_66, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_65, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_64, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1033, reinterpret_tensor(primals_238, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_236, (4096, 1024), (1024, 1), 0), buf1034, reinterpret_tensor(buf353, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf354, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf350, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf339, (16, 64, 512), (64, 1, 1024), 0), buf348, reinterpret_tensor(buf343, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf340, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf341, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf338, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf336, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_59, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_58, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_57, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1035, reinterpret_tensor(primals_230, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_228, (4096, 1024), (1024, 1), 0), buf1036, reinterpret_tensor(buf312, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf313, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf309, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf298, (16, 64, 512), (64, 1, 1024), 0), buf307, reinterpret_tensor(buf302, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf299, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf300, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf297, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf295, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_52, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_51, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_50, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1037, reinterpret_tensor(primals_222, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_220, (4096, 1024), (1024, 1), 0), buf1038, reinterpret_tensor(buf271, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf272, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf268, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf257, (16, 64, 512), (64, 1, 1024), 0), buf266, reinterpret_tensor(buf261, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf258, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf259, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf256, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf254, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_45, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_44, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_43, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1039, reinterpret_tensor(primals_214, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_212, (4096, 1024), (1024, 1), 0), buf1040, reinterpret_tensor(buf230, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf231, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf227, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf216, (16, 64, 512), (64, 1, 1024), 0), buf225, reinterpret_tensor(buf220, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf217, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf218, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf215, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf213, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_38, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_37, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_36, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1041, reinterpret_tensor(primals_206, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_204, (4096, 1024), (1024, 1), 0), buf1042, reinterpret_tensor(buf189, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf190, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf186, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf175, (16, 64, 512), (64, 1, 1024), 0), buf184, reinterpret_tensor(buf179, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf176, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf177, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf174, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf172, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_31, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_30, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_29, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1043, reinterpret_tensor(primals_198, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_196, (4096, 1024), (1024, 1), 0), buf1044, reinterpret_tensor(buf148, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf149, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf145, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf134, (16, 64, 512), (64, 1, 1024), 0), buf143, reinterpret_tensor(buf138, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf135, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf136, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf133, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf131, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_24, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_23, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_22, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1045, reinterpret_tensor(primals_190, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_188, (4096, 1024), (1024, 1), 0), buf1046, reinterpret_tensor(buf107, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf108, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf104, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf93, (16, 64, 512), (64, 1, 1024), 0), buf102, reinterpret_tensor(buf97, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf94, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf95, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf92, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf90, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_17, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_16, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_15, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1047, reinterpret_tensor(primals_182, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_180, (4096, 1024), (1024, 1), 0), buf1048, reinterpret_tensor(buf66, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf67, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf63, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf52, (16, 64, 512), (64, 1, 1024), 0), buf61, reinterpret_tensor(buf56, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf53, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf54, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf51, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf49, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_10, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_9, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_8, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1049, reinterpret_tensor(primals_174, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_172, (4096, 1024), (1024, 1), 0), buf1050, reinterpret_tensor(buf25, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf26, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf22, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf11, (16, 64, 512), (64, 1, 1024), 0), buf20, reinterpret_tensor(buf15, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf12, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf13, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf10, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf2, (1, 1024, 512), (524288, 1, 1024), 0), reinterpret_tensor(primals_3, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_2, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_1, (1, 1024, 1024), (1048576, 1, 1024), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((32000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_340 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_342 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_343 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_346 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_347 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_348 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_349 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_350 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_351 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_352 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_353 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_354 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_355 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_356 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_357 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_358 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_359 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_360 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_361 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_362 = rand_strided((32000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_363 = rand_strided((32000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_364 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_365 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('XLNetLMHeadModel', benchmark_compiled_module)
