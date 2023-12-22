
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


cpp_fused_add_1 = async_compile.cpp('''
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


cpp_fused_cat_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
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
                    out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp58;
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
    }
}
''')


cpp_fused__softmax_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_5 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_9 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_10 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_16 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_17 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_19 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_23 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_24 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_29 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_30 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_31 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_33 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_37 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_38 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
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


cpp_fused_add_native_layer_norm_41 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
    }
}
''')


cpp_fused__softmax_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_47 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_51 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_52 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_55 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_57 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_58 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_59 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_65 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_66 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_68 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_72 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_73 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_75 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_76 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_78 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_79 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_80 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
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


cpp_fused_add_native_layer_norm_83 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_85 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
    }
}
''')


cpp_fused__softmax_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_89 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_90 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_92 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_93 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_94 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_96 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_99 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_100 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_101 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_103 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_104 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_107 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_108 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_110 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_111 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_113 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_114 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_115 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_118 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_120 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_121 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_122 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
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


cpp_fused_add_native_layer_norm_125 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_127 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
    }
}
''')


cpp_fused__softmax_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_131 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_132 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_134 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_135 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_136 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_138 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_139 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_142 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_143 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_145 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_147 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_148 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_149 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_150 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_151 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_153 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_154 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_155 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_156 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_157 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_159 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_160 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_161 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_162 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_163 = async_compile.cpp('''
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


cpp_fused__softmax_add_index_select_mul_164 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_165 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_166 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_167 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_168 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_169 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_nll_loss_forward_170 = async_compile.cpp('''
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
        #pragma omp single
        {
            {
                {
                    float tmp_acc0 = 0;
                    long tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x0)];
                        auto tmp9 = out_ptr0[static_cast<long>(x0)];
                        auto tmp11 = out_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = static_cast<long>(0);
                        auto tmp4 = tmp2 ? tmp0 : tmp3;
                        auto tmp5 = decltype(tmp4)(tmp4 + 32000);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 32000L), "index out of bounds: 0 <= tmp7 < 32000L")
                        auto tmp8 = in_ptr0[static_cast<long>(tmp7 + (32000L*x0))];
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg1_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg2_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg3_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg4_1, (16, 64), (64, 1))
    assert_size_stride(arg5_1, (16, 64), (64, 1))
    assert_size_stride(arg6_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg7_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg8_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg9_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg10_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg11_1, (16, 64), (64, 1))
    assert_size_stride(arg12_1, (16, 64), (64, 1))
    assert_size_stride(arg13_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg14_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg15_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg16_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg17_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg18_1, (16, 64), (64, 1))
    assert_size_stride(arg19_1, (16, 64), (64, 1))
    assert_size_stride(arg20_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg21_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg22_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg23_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg24_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg25_1, (16, 64), (64, 1))
    assert_size_stride(arg26_1, (16, 64), (64, 1))
    assert_size_stride(arg27_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg28_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg29_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg30_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg31_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg32_1, (16, 64), (64, 1))
    assert_size_stride(arg33_1, (16, 64), (64, 1))
    assert_size_stride(arg34_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg35_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg36_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg37_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg38_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg39_1, (16, 64), (64, 1))
    assert_size_stride(arg40_1, (16, 64), (64, 1))
    assert_size_stride(arg41_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg42_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg43_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg44_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg45_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg46_1, (16, 64), (64, 1))
    assert_size_stride(arg47_1, (16, 64), (64, 1))
    assert_size_stride(arg48_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg49_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg50_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg51_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg52_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg53_1, (16, 64), (64, 1))
    assert_size_stride(arg54_1, (16, 64), (64, 1))
    assert_size_stride(arg55_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg56_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg57_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg58_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg59_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg60_1, (16, 64), (64, 1))
    assert_size_stride(arg61_1, (16, 64), (64, 1))
    assert_size_stride(arg62_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg63_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg64_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg65_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg66_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg67_1, (16, 64), (64, 1))
    assert_size_stride(arg68_1, (16, 64), (64, 1))
    assert_size_stride(arg69_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg70_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg71_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg72_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg73_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg74_1, (16, 64), (64, 1))
    assert_size_stride(arg75_1, (16, 64), (64, 1))
    assert_size_stride(arg76_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg77_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg78_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg79_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg80_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg81_1, (16, 64), (64, 1))
    assert_size_stride(arg82_1, (16, 64), (64, 1))
    assert_size_stride(arg83_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg84_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg85_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg86_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg87_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg88_1, (16, 64), (64, 1))
    assert_size_stride(arg89_1, (16, 64), (64, 1))
    assert_size_stride(arg90_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg91_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg92_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg93_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg94_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg95_1, (16, 64), (64, 1))
    assert_size_stride(arg96_1, (16, 64), (64, 1))
    assert_size_stride(arg97_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg98_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg99_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg100_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg101_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg102_1, (16, 64), (64, 1))
    assert_size_stride(arg103_1, (16, 64), (64, 1))
    assert_size_stride(arg104_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg105_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg106_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg107_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg108_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg109_1, (16, 64), (64, 1))
    assert_size_stride(arg110_1, (16, 64), (64, 1))
    assert_size_stride(arg111_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg112_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg113_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg114_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg115_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg116_1, (16, 64), (64, 1))
    assert_size_stride(arg117_1, (16, 64), (64, 1))
    assert_size_stride(arg118_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg119_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg120_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg121_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg122_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg123_1, (16, 64), (64, 1))
    assert_size_stride(arg124_1, (16, 64), (64, 1))
    assert_size_stride(arg125_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg126_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg127_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg128_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg129_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg130_1, (16, 64), (64, 1))
    assert_size_stride(arg131_1, (16, 64), (64, 1))
    assert_size_stride(arg132_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg133_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg134_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg135_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg136_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg137_1, (16, 64), (64, 1))
    assert_size_stride(arg138_1, (16, 64), (64, 1))
    assert_size_stride(arg139_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg140_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg141_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg142_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg143_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg144_1, (16, 64), (64, 1))
    assert_size_stride(arg145_1, (16, 64), (64, 1))
    assert_size_stride(arg146_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg147_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg148_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg149_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg150_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg151_1, (16, 64), (64, 1))
    assert_size_stride(arg152_1, (16, 64), (64, 1))
    assert_size_stride(arg153_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg154_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg155_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg156_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg157_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg158_1, (16, 64), (64, 1))
    assert_size_stride(arg159_1, (16, 64), (64, 1))
    assert_size_stride(arg160_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg161_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg162_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg163_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg164_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg165_1, (16, 64), (64, 1))
    assert_size_stride(arg166_1, (16, 64), (64, 1))
    assert_size_stride(arg167_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg168_1, (32000, 1024), (1024, 1))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg172_1, (4096, ), (1, ))
    assert_size_stride(arg173_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg174_1, (1024, ), (1, ))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (1024, ), (1, ))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg180_1, (4096, ), (1, ))
    assert_size_stride(arg181_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, ), (1, ))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, ), (1, ))
    assert_size_stride(arg186_1, (1024, ), (1, ))
    assert_size_stride(arg187_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg188_1, (4096, ), (1, ))
    assert_size_stride(arg189_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (1024, ), (1, ))
    assert_size_stride(arg193_1, (1024, ), (1, ))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg196_1, (4096, ), (1, ))
    assert_size_stride(arg197_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (1024, ), (1, ))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (1024, ), (1, ))
    assert_size_stride(arg202_1, (1024, ), (1, ))
    assert_size_stride(arg203_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg204_1, (4096, ), (1, ))
    assert_size_stride(arg205_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (1024, ), (1, ))
    assert_size_stride(arg208_1, (1024, ), (1, ))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (1024, ), (1, ))
    assert_size_stride(arg211_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg212_1, (4096, ), (1, ))
    assert_size_stride(arg213_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (1024, ), (1, ))
    assert_size_stride(arg216_1, (1024, ), (1, ))
    assert_size_stride(arg217_1, (1024, ), (1, ))
    assert_size_stride(arg218_1, (1024, ), (1, ))
    assert_size_stride(arg219_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg220_1, (4096, ), (1, ))
    assert_size_stride(arg221_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg222_1, (1024, ), (1, ))
    assert_size_stride(arg223_1, (1024, ), (1, ))
    assert_size_stride(arg224_1, (1024, ), (1, ))
    assert_size_stride(arg225_1, (1024, ), (1, ))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg228_1, (4096, ), (1, ))
    assert_size_stride(arg229_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg230_1, (1024, ), (1, ))
    assert_size_stride(arg231_1, (1024, ), (1, ))
    assert_size_stride(arg232_1, (1024, ), (1, ))
    assert_size_stride(arg233_1, (1024, ), (1, ))
    assert_size_stride(arg234_1, (1024, ), (1, ))
    assert_size_stride(arg235_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg236_1, (4096, ), (1, ))
    assert_size_stride(arg237_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg238_1, (1024, ), (1, ))
    assert_size_stride(arg239_1, (1024, ), (1, ))
    assert_size_stride(arg240_1, (1024, ), (1, ))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg244_1, (4096, ), (1, ))
    assert_size_stride(arg245_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg246_1, (1024, ), (1, ))
    assert_size_stride(arg247_1, (1024, ), (1, ))
    assert_size_stride(arg248_1, (1024, ), (1, ))
    assert_size_stride(arg249_1, (1024, ), (1, ))
    assert_size_stride(arg250_1, (1024, ), (1, ))
    assert_size_stride(arg251_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg252_1, (4096, ), (1, ))
    assert_size_stride(arg253_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg254_1, (1024, ), (1, ))
    assert_size_stride(arg255_1, (1024, ), (1, ))
    assert_size_stride(arg256_1, (1024, ), (1, ))
    assert_size_stride(arg257_1, (1024, ), (1, ))
    assert_size_stride(arg258_1, (1024, ), (1, ))
    assert_size_stride(arg259_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg260_1, (4096, ), (1, ))
    assert_size_stride(arg261_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg262_1, (1024, ), (1, ))
    assert_size_stride(arg263_1, (1024, ), (1, ))
    assert_size_stride(arg264_1, (1024, ), (1, ))
    assert_size_stride(arg265_1, (1024, ), (1, ))
    assert_size_stride(arg266_1, (1024, ), (1, ))
    assert_size_stride(arg267_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg268_1, (4096, ), (1, ))
    assert_size_stride(arg269_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg270_1, (1024, ), (1, ))
    assert_size_stride(arg271_1, (1024, ), (1, ))
    assert_size_stride(arg272_1, (1024, ), (1, ))
    assert_size_stride(arg273_1, (1024, ), (1, ))
    assert_size_stride(arg274_1, (1024, ), (1, ))
    assert_size_stride(arg275_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg276_1, (4096, ), (1, ))
    assert_size_stride(arg277_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg278_1, (1024, ), (1, ))
    assert_size_stride(arg279_1, (1024, ), (1, ))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (1024, ), (1, ))
    assert_size_stride(arg283_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg284_1, (4096, ), (1, ))
    assert_size_stride(arg285_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg286_1, (1024, ), (1, ))
    assert_size_stride(arg287_1, (1024, ), (1, ))
    assert_size_stride(arg288_1, (1024, ), (1, ))
    assert_size_stride(arg289_1, (1024, ), (1, ))
    assert_size_stride(arg290_1, (1024, ), (1, ))
    assert_size_stride(arg291_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg292_1, (4096, ), (1, ))
    assert_size_stride(arg293_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg294_1, (1024, ), (1, ))
    assert_size_stride(arg295_1, (1024, ), (1, ))
    assert_size_stride(arg296_1, (1024, ), (1, ))
    assert_size_stride(arg297_1, (1024, ), (1, ))
    assert_size_stride(arg298_1, (1024, ), (1, ))
    assert_size_stride(arg299_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg300_1, (4096, ), (1, ))
    assert_size_stride(arg301_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (1024, ), (1, ))
    assert_size_stride(arg304_1, (1024, ), (1, ))
    assert_size_stride(arg305_1, (1024, ), (1, ))
    assert_size_stride(arg306_1, (1024, ), (1, ))
    assert_size_stride(arg307_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg308_1, (4096, ), (1, ))
    assert_size_stride(arg309_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg310_1, (1024, ), (1, ))
    assert_size_stride(arg311_1, (1024, ), (1, ))
    assert_size_stride(arg312_1, (1024, ), (1, ))
    assert_size_stride(arg313_1, (1024, ), (1, ))
    assert_size_stride(arg314_1, (1024, ), (1, ))
    assert_size_stride(arg315_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg316_1, (4096, ), (1, ))
    assert_size_stride(arg317_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg318_1, (1024, ), (1, ))
    assert_size_stride(arg319_1, (1024, ), (1, ))
    assert_size_stride(arg320_1, (1024, ), (1, ))
    assert_size_stride(arg321_1, (1024, ), (1, ))
    assert_size_stride(arg322_1, (1024, ), (1, ))
    assert_size_stride(arg323_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg324_1, (4096, ), (1, ))
    assert_size_stride(arg325_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1024, ), (1, ))
    assert_size_stride(arg328_1, (1024, ), (1, ))
    assert_size_stride(arg329_1, (1024, ), (1, ))
    assert_size_stride(arg330_1, (1024, ), (1, ))
    assert_size_stride(arg331_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg332_1, (4096, ), (1, ))
    assert_size_stride(arg333_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg334_1, (1024, ), (1, ))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (1024, ), (1, ))
    assert_size_stride(arg337_1, (1024, ), (1, ))
    assert_size_stride(arg338_1, (1024, ), (1, ))
    assert_size_stride(arg339_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg340_1, (4096, ), (1, ))
    assert_size_stride(arg341_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1024, ), (1, ))
    assert_size_stride(arg344_1, (1024, ), (1, ))
    assert_size_stride(arg345_1, (1024, ), (1, ))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg348_1, (4096, ), (1, ))
    assert_size_stride(arg349_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg350_1, (1024, ), (1, ))
    assert_size_stride(arg351_1, (1024, ), (1, ))
    assert_size_stride(arg352_1, (1024, ), (1, ))
    assert_size_stride(arg353_1, (1024, ), (1, ))
    assert_size_stride(arg354_1, (1024, ), (1, ))
    assert_size_stride(arg355_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg356_1, (4096, ), (1, ))
    assert_size_stride(arg357_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg358_1, (1024, ), (1, ))
    assert_size_stride(arg359_1, (1024, ), (1, ))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (32000, 1024), (1024, 1))
    assert_size_stride(arg362_1, (32000, ), (1, ))
    assert_size_stride(arg363_1, (1, 512), (512, 1))
    assert_size_stride(arg364_1, (1, 512), (512, 1))
    buf0 = empty((512, 1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_0(c_void_p(arg363_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg168_1
    del arg363_1
    buf1 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [q_head_h], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf0, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(arg0_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf1)
    del arg0_1
    buf2 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_h], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf0, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg1_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf2)
    del arg1_1
    buf3 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    buf7 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_1(c_void_p(buf1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg4_1
    del arg5_1
    buf4 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [ac], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf3, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf2, (16, 64, 512), (64, 1, 1024), 0), out=buf4)
    buf5 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_cat_2(c_void_p(buf5.data_ptr()))
    buf6 = empty((1, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [k_head_r], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg3_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf6)
    del arg3_1
    buf8 = empty((16, 512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [bd], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf6, (16, 64, 1024), (64, 1, 1024), 0), out=buf8)
    buf9 = empty_strided((1, 16, 512, 1), (8192, 512, 1, 8192), device='cpu', dtype=torch.float32)
    buf10 = reinterpret_tensor(buf4, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf4  # reuse
    buf11 = empty_strided((1, 16, 512, 1), (8192, 512, 1, 8192), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_index_select_mul_3(c_void_p(buf10.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf11.data_ptr()))
    buf12 = reinterpret_tensor(buf7, (1, 512, 1024), (524288, 1024, 1), 0); del buf7  # reuse
    # Source Nodes: [v_head_h], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf0, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg2_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf12)
    del arg2_1
    buf13 = buf10; del buf10  # reuse
    cpp_fused__softmax_4(c_void_p(buf13.data_ptr()), c_void_p(buf11.data_ptr()))
    buf14 = reinterpret_tensor(buf3, (16, 512, 64), (32768, 64, 1), 0); del buf3  # reuse
    # Source Nodes: [attn_vec], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf13, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf12, (16, 512, 64), (64, 1024, 1), 0), out=buf14)
    buf15 = reinterpret_tensor(buf12, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf12  # reuse
    buf16 = reinterpret_tensor(buf6, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf6  # reuse
    cpp_fused_clone_5(c_void_p(buf14.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()))
    del arg6_1
    buf17 = reinterpret_tensor(buf14, (1, 512, 1024), (524288, 1024, 1), 0); del buf14  # reuse
    # Source Nodes: [attn_out], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf15, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf16, (1, 1024, 1024), (0, 1024, 1), 0), out=buf17)
    buf18 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((512, 1, 1), (1, 512, 512), device='cpu', dtype=torch.float32)
    buf21 = reinterpret_tensor(buf15, (512, 1, 1024), (1024, 1024, 1), 0); del buf15  # reuse
    cpp_fused_add_native_layer_norm_6(c_void_p(buf17.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()))
    del arg169_1
    del arg170_1
    buf22 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [output_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg172_1, reinterpret_tensor(buf21, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf22)
    del arg171_1
    del arg172_1
    buf23 = reinterpret_tensor(buf22, (512, 1, 4096), (4096, 4096, 1), 0); del buf22  # reuse
    cpp_fused_gelu_7(c_void_p(buf23.data_ptr()))
    buf24 = reinterpret_tensor(buf17, (512, 1024), (1024, 1), 0); del buf17  # reuse
    # Source Nodes: [output_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg174_1, reinterpret_tensor(buf23, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg173_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf24)
    del arg173_1
    del arg174_1
    buf25 = buf19; del buf19  # reuse
    buf26 = buf18; del buf18  # reuse
    buf28 = reinterpret_tensor(buf2, (512, 1, 1024), (1024, 1024, 1), 0); del buf2  # reuse
    cpp_fused_add_native_layer_norm_8(c_void_p(buf24.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf28.data_ptr()))
    del arg175_1
    del arg176_1
    buf29 = reinterpret_tensor(buf24, (1, 512, 1024), (524288, 1024, 1), 0); del buf24  # reuse
    # Source Nodes: [q_head_h_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf28, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg7_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf29)
    del arg7_1
    buf30 = reinterpret_tensor(buf21, (1, 512, 1024), (524288, 1024, 1), 0); del buf21  # reuse
    # Source Nodes: [k_head_h_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf28, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg8_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf30)
    del arg8_1
    buf31 = reinterpret_tensor(buf1, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf1  # reuse
    buf34 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_9(c_void_p(buf29.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf34.data_ptr()))
    del arg11_1
    del arg12_1
    buf32 = reinterpret_tensor(buf13, (16, 512, 512), (262144, 512, 1), 0); del buf13  # reuse
    # Source Nodes: [ac_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf31, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf30, (16, 64, 512), (64, 1, 1024), 0), out=buf32)
    buf33 = reinterpret_tensor(buf16, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf16  # reuse
    # Source Nodes: [k_head_r_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg10_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf33)
    del arg10_1
    buf35 = buf8; del buf8  # reuse
    # Source Nodes: [bd_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf34, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf33, (16, 64, 1024), (64, 1, 1024), 0), out=buf35)
    buf36 = buf11; del buf11  # reuse
    buf37 = reinterpret_tensor(buf32, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf32  # reuse
    buf38 = buf9; del buf9  # reuse
    cpp_fused__softmax_add_index_select_mul_10(c_void_p(buf37.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()))
    buf39 = reinterpret_tensor(buf34, (1, 512, 1024), (524288, 1024, 1), 0); del buf34  # reuse
    # Source Nodes: [v_head_h_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf28, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg9_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf39)
    del arg9_1
    buf40 = buf37; del buf37  # reuse
    cpp_fused__softmax_11(c_void_p(buf40.data_ptr()), c_void_p(buf38.data_ptr()))
    buf41 = reinterpret_tensor(buf31, (16, 512, 64), (32768, 64, 1), 0); del buf31  # reuse
    # Source Nodes: [attn_vec_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf40, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf39, (16, 512, 64), (64, 1024, 1), 0), out=buf41)
    buf42 = reinterpret_tensor(buf39, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf39  # reuse
    buf43 = reinterpret_tensor(buf33, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf33  # reuse
    cpp_fused_clone_12(c_void_p(buf41.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    del arg13_1
    buf44 = reinterpret_tensor(buf41, (1, 512, 1024), (524288, 1024, 1), 0); del buf41  # reuse
    # Source Nodes: [attn_out_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf42, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf43, (1, 1024, 1024), (0, 1024, 1), 0), out=buf44)
    buf45 = buf26; del buf26  # reuse
    buf46 = buf25; del buf25  # reuse
    buf48 = reinterpret_tensor(buf42, (512, 1, 1024), (1024, 1024, 1), 0); del buf42  # reuse
    cpp_fused_add_native_layer_norm_13(c_void_p(buf44.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()))
    del arg177_1
    del arg178_1
    buf49 = reinterpret_tensor(buf23, (512, 4096), (4096, 1), 0); del buf23  # reuse
    # Source Nodes: [output_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg180_1, reinterpret_tensor(buf48, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg179_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf49)
    del arg179_1
    del arg180_1
    buf50 = reinterpret_tensor(buf49, (512, 1, 4096), (4096, 4096, 1), 0); del buf49  # reuse
    cpp_fused_gelu_14(c_void_p(buf50.data_ptr()))
    buf51 = reinterpret_tensor(buf44, (512, 1024), (1024, 1), 0); del buf44  # reuse
    # Source Nodes: [output_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg182_1, reinterpret_tensor(buf50, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg181_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf51)
    del arg181_1
    del arg182_1
    buf52 = buf46; del buf46  # reuse
    buf53 = buf45; del buf45  # reuse
    buf55 = reinterpret_tensor(buf30, (512, 1, 1024), (1024, 1024, 1), 0); del buf30  # reuse
    cpp_fused_add_native_layer_norm_15(c_void_p(buf51.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()))
    del arg183_1
    del arg184_1
    buf56 = reinterpret_tensor(buf51, (1, 512, 1024), (524288, 1024, 1), 0); del buf51  # reuse
    # Source Nodes: [q_head_h_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf55, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg14_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf56)
    del arg14_1
    buf57 = reinterpret_tensor(buf48, (1, 512, 1024), (524288, 1024, 1), 0); del buf48  # reuse
    # Source Nodes: [k_head_h_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf55, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg15_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf57)
    del arg15_1
    buf58 = reinterpret_tensor(buf29, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf29  # reuse
    buf61 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_16(c_void_p(buf56.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf61.data_ptr()))
    del arg18_1
    del arg19_1
    buf59 = reinterpret_tensor(buf40, (16, 512, 512), (262144, 512, 1), 0); del buf40  # reuse
    # Source Nodes: [ac_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf58, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf57, (16, 64, 512), (64, 1, 1024), 0), out=buf59)
    buf60 = reinterpret_tensor(buf43, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf43  # reuse
    # Source Nodes: [k_head_r_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg17_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf60)
    del arg17_1
    buf62 = buf35; del buf35  # reuse
    # Source Nodes: [bd_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf61, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf60, (16, 64, 1024), (64, 1, 1024), 0), out=buf62)
    buf63 = buf38; del buf38  # reuse
    buf64 = reinterpret_tensor(buf59, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf59  # reuse
    buf65 = buf36; del buf36  # reuse
    cpp_fused__softmax_add_index_select_mul_17(c_void_p(buf64.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf65.data_ptr()))
    buf66 = reinterpret_tensor(buf61, (1, 512, 1024), (524288, 1024, 1), 0); del buf61  # reuse
    # Source Nodes: [v_head_h_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf55, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg16_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf66)
    del arg16_1
    buf67 = buf64; del buf64  # reuse
    cpp_fused__softmax_18(c_void_p(buf67.data_ptr()), c_void_p(buf65.data_ptr()))
    buf68 = reinterpret_tensor(buf58, (16, 512, 64), (32768, 64, 1), 0); del buf58  # reuse
    # Source Nodes: [attn_vec_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf67, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf66, (16, 512, 64), (64, 1024, 1), 0), out=buf68)
    buf69 = reinterpret_tensor(buf66, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf66  # reuse
    buf70 = reinterpret_tensor(buf60, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf60  # reuse
    cpp_fused_clone_19(c_void_p(buf68.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del arg20_1
    buf71 = reinterpret_tensor(buf68, (1, 512, 1024), (524288, 1024, 1), 0); del buf68  # reuse
    # Source Nodes: [attn_out_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf69, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf70, (1, 1024, 1024), (0, 1024, 1), 0), out=buf71)
    buf72 = buf53; del buf53  # reuse
    buf73 = buf52; del buf52  # reuse
    buf75 = reinterpret_tensor(buf69, (512, 1, 1024), (1024, 1024, 1), 0); del buf69  # reuse
    cpp_fused_add_native_layer_norm_20(c_void_p(buf71.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()))
    del arg185_1
    del arg186_1
    buf76 = reinterpret_tensor(buf50, (512, 4096), (4096, 1), 0); del buf50  # reuse
    # Source Nodes: [output_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg188_1, reinterpret_tensor(buf75, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg187_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf76)
    del arg187_1
    del arg188_1
    buf77 = reinterpret_tensor(buf76, (512, 1, 4096), (4096, 4096, 1), 0); del buf76  # reuse
    cpp_fused_gelu_21(c_void_p(buf77.data_ptr()))
    buf78 = reinterpret_tensor(buf71, (512, 1024), (1024, 1), 0); del buf71  # reuse
    # Source Nodes: [output_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg190_1, reinterpret_tensor(buf77, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg189_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf78)
    del arg189_1
    del arg190_1
    buf79 = buf73; del buf73  # reuse
    buf80 = buf72; del buf72  # reuse
    buf82 = reinterpret_tensor(buf57, (512, 1, 1024), (1024, 1024, 1), 0); del buf57  # reuse
    cpp_fused_add_native_layer_norm_22(c_void_p(buf78.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()))
    del arg191_1
    del arg192_1
    buf83 = reinterpret_tensor(buf78, (1, 512, 1024), (524288, 1024, 1), 0); del buf78  # reuse
    # Source Nodes: [q_head_h_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf82, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg21_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf83)
    del arg21_1
    buf84 = reinterpret_tensor(buf75, (1, 512, 1024), (524288, 1024, 1), 0); del buf75  # reuse
    # Source Nodes: [k_head_h_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf82, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg22_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf84)
    del arg22_1
    buf85 = reinterpret_tensor(buf56, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf56  # reuse
    buf88 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_23(c_void_p(buf83.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf88.data_ptr()))
    del arg25_1
    del arg26_1
    buf86 = reinterpret_tensor(buf67, (16, 512, 512), (262144, 512, 1), 0); del buf67  # reuse
    # Source Nodes: [ac_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf85, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf84, (16, 64, 512), (64, 1, 1024), 0), out=buf86)
    buf87 = reinterpret_tensor(buf70, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf70  # reuse
    # Source Nodes: [k_head_r_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg24_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf87)
    del arg24_1
    buf89 = buf62; del buf62  # reuse
    # Source Nodes: [bd_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf88, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf87, (16, 64, 1024), (64, 1, 1024), 0), out=buf89)
    buf90 = buf65; del buf65  # reuse
    buf91 = reinterpret_tensor(buf86, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf86  # reuse
    buf92 = buf63; del buf63  # reuse
    cpp_fused__softmax_add_index_select_mul_24(c_void_p(buf91.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf92.data_ptr()))
    buf93 = reinterpret_tensor(buf88, (1, 512, 1024), (524288, 1024, 1), 0); del buf88  # reuse
    # Source Nodes: [v_head_h_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf82, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg23_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf93)
    del arg23_1
    buf94 = buf91; del buf91  # reuse
    cpp_fused__softmax_25(c_void_p(buf94.data_ptr()), c_void_p(buf92.data_ptr()))
    buf95 = reinterpret_tensor(buf85, (16, 512, 64), (32768, 64, 1), 0); del buf85  # reuse
    # Source Nodes: [attn_vec_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf94, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf93, (16, 512, 64), (64, 1024, 1), 0), out=buf95)
    buf96 = reinterpret_tensor(buf93, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf93  # reuse
    buf97 = reinterpret_tensor(buf87, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf87  # reuse
    cpp_fused_clone_26(c_void_p(buf95.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del arg27_1
    buf98 = reinterpret_tensor(buf95, (1, 512, 1024), (524288, 1024, 1), 0); del buf95  # reuse
    # Source Nodes: [attn_out_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf96, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf97, (1, 1024, 1024), (0, 1024, 1), 0), out=buf98)
    buf99 = buf80; del buf80  # reuse
    buf100 = buf79; del buf79  # reuse
    buf102 = reinterpret_tensor(buf96, (512, 1, 1024), (1024, 1024, 1), 0); del buf96  # reuse
    cpp_fused_add_native_layer_norm_27(c_void_p(buf98.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()))
    del arg193_1
    del arg194_1
    buf103 = reinterpret_tensor(buf77, (512, 4096), (4096, 1), 0); del buf77  # reuse
    # Source Nodes: [output_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg196_1, reinterpret_tensor(buf102, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg195_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf103)
    del arg195_1
    del arg196_1
    buf104 = reinterpret_tensor(buf103, (512, 1, 4096), (4096, 4096, 1), 0); del buf103  # reuse
    cpp_fused_gelu_28(c_void_p(buf104.data_ptr()))
    buf105 = reinterpret_tensor(buf98, (512, 1024), (1024, 1), 0); del buf98  # reuse
    # Source Nodes: [output_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg198_1, reinterpret_tensor(buf104, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg197_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf105)
    del arg197_1
    del arg198_1
    buf106 = buf99; del buf99  # reuse
    buf107 = buf100; del buf100  # reuse
    buf109 = reinterpret_tensor(buf84, (512, 1, 1024), (1024, 1024, 1), 0); del buf84  # reuse
    cpp_fused_add_native_layer_norm_29(c_void_p(buf105.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()))
    del arg199_1
    del arg200_1
    buf110 = reinterpret_tensor(buf105, (1, 512, 1024), (524288, 1024, 1), 0); del buf105  # reuse
    # Source Nodes: [q_head_h_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf109, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg28_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf110)
    del arg28_1
    buf111 = reinterpret_tensor(buf102, (1, 512, 1024), (524288, 1024, 1), 0); del buf102  # reuse
    # Source Nodes: [k_head_h_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf109, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg29_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf111)
    del arg29_1
    buf112 = reinterpret_tensor(buf83, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf83  # reuse
    buf115 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_30(c_void_p(buf110.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf115.data_ptr()))
    del arg32_1
    del arg33_1
    buf113 = reinterpret_tensor(buf94, (16, 512, 512), (262144, 512, 1), 0); del buf94  # reuse
    # Source Nodes: [ac_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf112, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf111, (16, 64, 512), (64, 1, 1024), 0), out=buf113)
    buf114 = reinterpret_tensor(buf97, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf97  # reuse
    # Source Nodes: [k_head_r_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg31_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf114)
    del arg31_1
    buf116 = buf89; del buf89  # reuse
    # Source Nodes: [bd_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf115, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf114, (16, 64, 1024), (64, 1, 1024), 0), out=buf116)
    buf117 = buf92; del buf92  # reuse
    buf118 = reinterpret_tensor(buf113, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf113  # reuse
    buf119 = buf90; del buf90  # reuse
    cpp_fused__softmax_add_index_select_mul_31(c_void_p(buf118.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()))
    buf120 = reinterpret_tensor(buf115, (1, 512, 1024), (524288, 1024, 1), 0); del buf115  # reuse
    # Source Nodes: [v_head_h_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf109, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg30_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf120)
    del arg30_1
    buf121 = buf118; del buf118  # reuse
    cpp_fused__softmax_32(c_void_p(buf121.data_ptr()), c_void_p(buf119.data_ptr()))
    buf122 = reinterpret_tensor(buf112, (16, 512, 64), (32768, 64, 1), 0); del buf112  # reuse
    # Source Nodes: [attn_vec_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf121, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf120, (16, 512, 64), (64, 1024, 1), 0), out=buf122)
    buf123 = reinterpret_tensor(buf120, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf120  # reuse
    buf124 = reinterpret_tensor(buf114, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf114  # reuse
    cpp_fused_clone_33(c_void_p(buf122.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()))
    del arg34_1
    buf125 = reinterpret_tensor(buf122, (1, 512, 1024), (524288, 1024, 1), 0); del buf122  # reuse
    # Source Nodes: [attn_out_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf123, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf124, (1, 1024, 1024), (0, 1024, 1), 0), out=buf125)
    buf126 = buf107; del buf107  # reuse
    buf127 = buf106; del buf106  # reuse
    buf129 = reinterpret_tensor(buf123, (512, 1, 1024), (1024, 1024, 1), 0); del buf123  # reuse
    cpp_fused_add_native_layer_norm_34(c_void_p(buf125.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()))
    del arg201_1
    del arg202_1
    buf130 = reinterpret_tensor(buf104, (512, 4096), (4096, 1), 0); del buf104  # reuse
    # Source Nodes: [output_34], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg204_1, reinterpret_tensor(buf129, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg203_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf130)
    del arg203_1
    del arg204_1
    buf131 = reinterpret_tensor(buf130, (512, 1, 4096), (4096, 4096, 1), 0); del buf130  # reuse
    cpp_fused_gelu_35(c_void_p(buf131.data_ptr()))
    buf132 = reinterpret_tensor(buf125, (512, 1024), (1024, 1), 0); del buf125  # reuse
    # Source Nodes: [output_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg206_1, reinterpret_tensor(buf131, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg205_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf132)
    del arg205_1
    del arg206_1
    buf133 = buf127; del buf127  # reuse
    buf134 = buf126; del buf126  # reuse
    buf136 = reinterpret_tensor(buf111, (512, 1, 1024), (1024, 1024, 1), 0); del buf111  # reuse
    cpp_fused_add_native_layer_norm_36(c_void_p(buf132.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()))
    del arg207_1
    del arg208_1
    buf137 = reinterpret_tensor(buf132, (1, 512, 1024), (524288, 1024, 1), 0); del buf132  # reuse
    # Source Nodes: [q_head_h_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf136, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg35_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf137)
    del arg35_1
    buf138 = reinterpret_tensor(buf129, (1, 512, 1024), (524288, 1024, 1), 0); del buf129  # reuse
    # Source Nodes: [k_head_h_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf136, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg36_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf138)
    del arg36_1
    buf139 = reinterpret_tensor(buf110, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf110  # reuse
    buf142 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_37(c_void_p(buf137.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf142.data_ptr()))
    del arg39_1
    del arg40_1
    buf140 = reinterpret_tensor(buf121, (16, 512, 512), (262144, 512, 1), 0); del buf121  # reuse
    # Source Nodes: [ac_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf139, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf138, (16, 64, 512), (64, 1, 1024), 0), out=buf140)
    buf141 = reinterpret_tensor(buf124, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf124  # reuse
    # Source Nodes: [k_head_r_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg38_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf141)
    del arg38_1
    buf143 = buf116; del buf116  # reuse
    # Source Nodes: [bd_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf142, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf141, (16, 64, 1024), (64, 1, 1024), 0), out=buf143)
    buf144 = buf119; del buf119  # reuse
    buf145 = reinterpret_tensor(buf140, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf140  # reuse
    buf146 = buf117; del buf117  # reuse
    cpp_fused__softmax_add_index_select_mul_38(c_void_p(buf145.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()))
    buf147 = reinterpret_tensor(buf142, (1, 512, 1024), (524288, 1024, 1), 0); del buf142  # reuse
    # Source Nodes: [v_head_h_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf136, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg37_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf147)
    del arg37_1
    buf148 = buf145; del buf145  # reuse
    cpp_fused__softmax_39(c_void_p(buf148.data_ptr()), c_void_p(buf146.data_ptr()))
    buf149 = reinterpret_tensor(buf139, (16, 512, 64), (32768, 64, 1), 0); del buf139  # reuse
    # Source Nodes: [attn_vec_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf148, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf147, (16, 512, 64), (64, 1024, 1), 0), out=buf149)
    buf150 = reinterpret_tensor(buf147, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf147  # reuse
    buf151 = reinterpret_tensor(buf141, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf141  # reuse
    cpp_fused_clone_40(c_void_p(buf149.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()))
    del arg41_1
    buf152 = reinterpret_tensor(buf149, (1, 512, 1024), (524288, 1024, 1), 0); del buf149  # reuse
    # Source Nodes: [attn_out_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf150, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf151, (1, 1024, 1024), (0, 1024, 1), 0), out=buf152)
    buf153 = buf134; del buf134  # reuse
    buf154 = buf133; del buf133  # reuse
    buf156 = reinterpret_tensor(buf150, (512, 1, 1024), (1024, 1024, 1), 0); del buf150  # reuse
    cpp_fused_add_native_layer_norm_41(c_void_p(buf152.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()))
    del arg209_1
    del arg210_1
    buf157 = reinterpret_tensor(buf131, (512, 4096), (4096, 1), 0); del buf131  # reuse
    # Source Nodes: [output_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg212_1, reinterpret_tensor(buf156, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg211_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf157)
    del arg211_1
    del arg212_1
    buf158 = reinterpret_tensor(buf157, (512, 1, 4096), (4096, 4096, 1), 0); del buf157  # reuse
    cpp_fused_gelu_42(c_void_p(buf158.data_ptr()))
    buf159 = reinterpret_tensor(buf152, (512, 1024), (1024, 1), 0); del buf152  # reuse
    # Source Nodes: [output_45], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg214_1, reinterpret_tensor(buf158, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg213_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf159)
    del arg213_1
    del arg214_1
    buf160 = buf154; del buf154  # reuse
    buf161 = buf153; del buf153  # reuse
    buf163 = reinterpret_tensor(buf138, (512, 1, 1024), (1024, 1024, 1), 0); del buf138  # reuse
    cpp_fused_add_native_layer_norm_43(c_void_p(buf159.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()))
    del arg215_1
    del arg216_1
    buf164 = reinterpret_tensor(buf159, (1, 512, 1024), (524288, 1024, 1), 0); del buf159  # reuse
    # Source Nodes: [q_head_h_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf163, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg42_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf164)
    del arg42_1
    buf165 = reinterpret_tensor(buf156, (1, 512, 1024), (524288, 1024, 1), 0); del buf156  # reuse
    # Source Nodes: [k_head_h_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf163, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg43_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf165)
    del arg43_1
    buf166 = reinterpret_tensor(buf137, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf137  # reuse
    buf169 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_44(c_void_p(buf164.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf169.data_ptr()))
    del arg46_1
    del arg47_1
    buf167 = reinterpret_tensor(buf148, (16, 512, 512), (262144, 512, 1), 0); del buf148  # reuse
    # Source Nodes: [ac_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf166, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf165, (16, 64, 512), (64, 1, 1024), 0), out=buf167)
    buf168 = reinterpret_tensor(buf151, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf151  # reuse
    # Source Nodes: [k_head_r_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg45_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf168)
    del arg45_1
    buf170 = buf143; del buf143  # reuse
    # Source Nodes: [bd_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf169, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf168, (16, 64, 1024), (64, 1, 1024), 0), out=buf170)
    buf171 = buf146; del buf146  # reuse
    buf172 = reinterpret_tensor(buf167, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf167  # reuse
    buf173 = buf144; del buf144  # reuse
    cpp_fused__softmax_add_index_select_mul_45(c_void_p(buf172.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()))
    buf174 = reinterpret_tensor(buf169, (1, 512, 1024), (524288, 1024, 1), 0); del buf169  # reuse
    # Source Nodes: [v_head_h_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf163, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg44_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf174)
    del arg44_1
    buf175 = buf172; del buf172  # reuse
    cpp_fused__softmax_46(c_void_p(buf175.data_ptr()), c_void_p(buf173.data_ptr()))
    buf176 = reinterpret_tensor(buf166, (16, 512, 64), (32768, 64, 1), 0); del buf166  # reuse
    # Source Nodes: [attn_vec_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf175, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf174, (16, 512, 64), (64, 1024, 1), 0), out=buf176)
    buf177 = reinterpret_tensor(buf174, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf174  # reuse
    buf178 = reinterpret_tensor(buf168, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf168  # reuse
    cpp_fused_clone_47(c_void_p(buf176.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()))
    del arg48_1
    buf179 = reinterpret_tensor(buf176, (1, 512, 1024), (524288, 1024, 1), 0); del buf176  # reuse
    # Source Nodes: [attn_out_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf177, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf178, (1, 1024, 1024), (0, 1024, 1), 0), out=buf179)
    buf180 = buf161; del buf161  # reuse
    buf181 = buf160; del buf160  # reuse
    buf183 = reinterpret_tensor(buf177, (512, 1, 1024), (1024, 1024, 1), 0); del buf177  # reuse
    cpp_fused_add_native_layer_norm_48(c_void_p(buf179.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()))
    del arg217_1
    del arg218_1
    buf184 = reinterpret_tensor(buf158, (512, 4096), (4096, 1), 0); del buf158  # reuse
    # Source Nodes: [output_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg220_1, reinterpret_tensor(buf183, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg219_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf184)
    del arg219_1
    del arg220_1
    buf185 = reinterpret_tensor(buf184, (512, 1, 4096), (4096, 4096, 1), 0); del buf184  # reuse
    cpp_fused_gelu_49(c_void_p(buf185.data_ptr()))
    buf186 = reinterpret_tensor(buf179, (512, 1024), (1024, 1), 0); del buf179  # reuse
    # Source Nodes: [output_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg222_1, reinterpret_tensor(buf185, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg221_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf186)
    del arg221_1
    del arg222_1
    buf187 = buf181; del buf181  # reuse
    buf188 = buf180; del buf180  # reuse
    buf190 = reinterpret_tensor(buf165, (512, 1, 1024), (1024, 1024, 1), 0); del buf165  # reuse
    cpp_fused_add_native_layer_norm_50(c_void_p(buf186.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf190.data_ptr()))
    del arg223_1
    del arg224_1
    buf191 = reinterpret_tensor(buf186, (1, 512, 1024), (524288, 1024, 1), 0); del buf186  # reuse
    # Source Nodes: [q_head_h_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf190, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg49_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf191)
    del arg49_1
    buf192 = reinterpret_tensor(buf183, (1, 512, 1024), (524288, 1024, 1), 0); del buf183  # reuse
    # Source Nodes: [k_head_h_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf190, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg50_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf192)
    del arg50_1
    buf193 = reinterpret_tensor(buf164, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf164  # reuse
    buf196 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_51(c_void_p(buf191.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf196.data_ptr()))
    del arg53_1
    del arg54_1
    buf194 = reinterpret_tensor(buf175, (16, 512, 512), (262144, 512, 1), 0); del buf175  # reuse
    # Source Nodes: [ac_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf193, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf192, (16, 64, 512), (64, 1, 1024), 0), out=buf194)
    buf195 = reinterpret_tensor(buf178, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf178  # reuse
    # Source Nodes: [k_head_r_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg52_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf195)
    del arg52_1
    buf197 = buf170; del buf170  # reuse
    # Source Nodes: [bd_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf196, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf195, (16, 64, 1024), (64, 1, 1024), 0), out=buf197)
    buf198 = buf173; del buf173  # reuse
    buf199 = reinterpret_tensor(buf194, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf194  # reuse
    buf200 = buf171; del buf171  # reuse
    cpp_fused__softmax_add_index_select_mul_52(c_void_p(buf199.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf200.data_ptr()))
    buf201 = reinterpret_tensor(buf196, (1, 512, 1024), (524288, 1024, 1), 0); del buf196  # reuse
    # Source Nodes: [v_head_h_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf190, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg51_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf201)
    del arg51_1
    buf202 = buf199; del buf199  # reuse
    cpp_fused__softmax_53(c_void_p(buf202.data_ptr()), c_void_p(buf200.data_ptr()))
    buf203 = reinterpret_tensor(buf193, (16, 512, 64), (32768, 64, 1), 0); del buf193  # reuse
    # Source Nodes: [attn_vec_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf202, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf201, (16, 512, 64), (64, 1024, 1), 0), out=buf203)
    buf204 = reinterpret_tensor(buf201, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf201  # reuse
    buf205 = reinterpret_tensor(buf195, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf195  # reuse
    cpp_fused_clone_54(c_void_p(buf203.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()))
    del arg55_1
    buf206 = reinterpret_tensor(buf203, (1, 512, 1024), (524288, 1024, 1), 0); del buf203  # reuse
    # Source Nodes: [attn_out_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf204, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf205, (1, 1024, 1024), (0, 1024, 1), 0), out=buf206)
    buf207 = buf188; del buf188  # reuse
    buf208 = buf187; del buf187  # reuse
    buf210 = reinterpret_tensor(buf204, (512, 1, 1024), (1024, 1024, 1), 0); del buf204  # reuse
    cpp_fused_add_native_layer_norm_55(c_void_p(buf206.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()))
    del arg225_1
    del arg226_1
    buf211 = reinterpret_tensor(buf185, (512, 4096), (4096, 1), 0); del buf185  # reuse
    # Source Nodes: [output_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg228_1, reinterpret_tensor(buf210, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg227_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf211)
    del arg227_1
    del arg228_1
    buf212 = reinterpret_tensor(buf211, (512, 1, 4096), (4096, 4096, 1), 0); del buf211  # reuse
    cpp_fused_gelu_56(c_void_p(buf212.data_ptr()))
    buf213 = reinterpret_tensor(buf206, (512, 1024), (1024, 1), 0); del buf206  # reuse
    # Source Nodes: [output_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg230_1, reinterpret_tensor(buf212, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg229_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf213)
    del arg229_1
    del arg230_1
    buf214 = buf208; del buf208  # reuse
    buf215 = buf207; del buf207  # reuse
    buf217 = reinterpret_tensor(buf192, (512, 1, 1024), (1024, 1024, 1), 0); del buf192  # reuse
    cpp_fused_add_native_layer_norm_57(c_void_p(buf213.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf217.data_ptr()))
    del arg231_1
    del arg232_1
    buf218 = reinterpret_tensor(buf213, (1, 512, 1024), (524288, 1024, 1), 0); del buf213  # reuse
    # Source Nodes: [q_head_h_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf217, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg56_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf218)
    del arg56_1
    buf219 = reinterpret_tensor(buf210, (1, 512, 1024), (524288, 1024, 1), 0); del buf210  # reuse
    # Source Nodes: [k_head_h_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf217, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg57_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf219)
    del arg57_1
    buf220 = reinterpret_tensor(buf191, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf191  # reuse
    buf223 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_58(c_void_p(buf218.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf223.data_ptr()))
    del arg60_1
    del arg61_1
    buf221 = reinterpret_tensor(buf202, (16, 512, 512), (262144, 512, 1), 0); del buf202  # reuse
    # Source Nodes: [ac_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf220, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf219, (16, 64, 512), (64, 1, 1024), 0), out=buf221)
    buf222 = reinterpret_tensor(buf205, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf205  # reuse
    # Source Nodes: [k_head_r_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg59_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf222)
    del arg59_1
    buf224 = buf197; del buf197  # reuse
    # Source Nodes: [bd_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf223, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf222, (16, 64, 1024), (64, 1, 1024), 0), out=buf224)
    buf225 = buf200; del buf200  # reuse
    buf226 = reinterpret_tensor(buf221, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf221  # reuse
    buf227 = buf198; del buf198  # reuse
    cpp_fused__softmax_add_index_select_mul_59(c_void_p(buf226.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf227.data_ptr()))
    buf228 = reinterpret_tensor(buf223, (1, 512, 1024), (524288, 1024, 1), 0); del buf223  # reuse
    # Source Nodes: [v_head_h_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf217, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg58_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf228)
    del arg58_1
    buf229 = buf226; del buf226  # reuse
    cpp_fused__softmax_60(c_void_p(buf229.data_ptr()), c_void_p(buf227.data_ptr()))
    buf230 = reinterpret_tensor(buf220, (16, 512, 64), (32768, 64, 1), 0); del buf220  # reuse
    # Source Nodes: [attn_vec_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf229, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf228, (16, 512, 64), (64, 1024, 1), 0), out=buf230)
    buf231 = reinterpret_tensor(buf228, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf228  # reuse
    buf232 = reinterpret_tensor(buf222, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf222  # reuse
    cpp_fused_clone_61(c_void_p(buf230.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    del arg62_1
    buf233 = reinterpret_tensor(buf230, (1, 512, 1024), (524288, 1024, 1), 0); del buf230  # reuse
    # Source Nodes: [attn_out_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf231, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf232, (1, 1024, 1024), (0, 1024, 1), 0), out=buf233)
    buf234 = buf215; del buf215  # reuse
    buf235 = buf214; del buf214  # reuse
    buf237 = reinterpret_tensor(buf231, (512, 1, 1024), (1024, 1024, 1), 0); del buf231  # reuse
    cpp_fused_add_native_layer_norm_62(c_void_p(buf233.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf237.data_ptr()))
    del arg233_1
    del arg234_1
    buf238 = reinterpret_tensor(buf212, (512, 4096), (4096, 1), 0); del buf212  # reuse
    # Source Nodes: [output_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg236_1, reinterpret_tensor(buf237, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg235_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf238)
    del arg235_1
    del arg236_1
    buf239 = reinterpret_tensor(buf238, (512, 1, 4096), (4096, 4096, 1), 0); del buf238  # reuse
    cpp_fused_gelu_63(c_void_p(buf239.data_ptr()))
    buf240 = reinterpret_tensor(buf233, (512, 1024), (1024, 1), 0); del buf233  # reuse
    # Source Nodes: [output_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg238_1, reinterpret_tensor(buf239, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg237_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf240)
    del arg237_1
    del arg238_1
    buf241 = buf235; del buf235  # reuse
    buf242 = buf234; del buf234  # reuse
    buf244 = reinterpret_tensor(buf219, (512, 1, 1024), (1024, 1024, 1), 0); del buf219  # reuse
    cpp_fused_add_native_layer_norm_64(c_void_p(buf240.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf244.data_ptr()))
    del arg239_1
    del arg240_1
    buf245 = reinterpret_tensor(buf240, (1, 512, 1024), (524288, 1024, 1), 0); del buf240  # reuse
    # Source Nodes: [q_head_h_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf244, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg63_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf245)
    del arg63_1
    buf246 = reinterpret_tensor(buf237, (1, 512, 1024), (524288, 1024, 1), 0); del buf237  # reuse
    # Source Nodes: [k_head_h_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf244, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg64_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf246)
    del arg64_1
    buf247 = reinterpret_tensor(buf218, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf218  # reuse
    buf250 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_65(c_void_p(buf245.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf250.data_ptr()))
    del arg67_1
    del arg68_1
    buf248 = reinterpret_tensor(buf229, (16, 512, 512), (262144, 512, 1), 0); del buf229  # reuse
    # Source Nodes: [ac_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf247, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf246, (16, 64, 512), (64, 1, 1024), 0), out=buf248)
    buf249 = reinterpret_tensor(buf232, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf232  # reuse
    # Source Nodes: [k_head_r_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg66_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf249)
    del arg66_1
    buf251 = buf224; del buf224  # reuse
    # Source Nodes: [bd_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf250, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf249, (16, 64, 1024), (64, 1, 1024), 0), out=buf251)
    buf252 = buf227; del buf227  # reuse
    buf253 = reinterpret_tensor(buf248, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf248  # reuse
    buf254 = buf225; del buf225  # reuse
    cpp_fused__softmax_add_index_select_mul_66(c_void_p(buf253.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf254.data_ptr()))
    buf255 = reinterpret_tensor(buf250, (1, 512, 1024), (524288, 1024, 1), 0); del buf250  # reuse
    # Source Nodes: [v_head_h_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf244, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg65_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf255)
    del arg65_1
    buf256 = buf253; del buf253  # reuse
    cpp_fused__softmax_67(c_void_p(buf256.data_ptr()), c_void_p(buf254.data_ptr()))
    buf257 = reinterpret_tensor(buf247, (16, 512, 64), (32768, 64, 1), 0); del buf247  # reuse
    # Source Nodes: [attn_vec_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf256, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf255, (16, 512, 64), (64, 1024, 1), 0), out=buf257)
    buf258 = reinterpret_tensor(buf255, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf255  # reuse
    buf259 = reinterpret_tensor(buf249, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf249  # reuse
    cpp_fused_clone_68(c_void_p(buf257.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()))
    del arg69_1
    buf260 = reinterpret_tensor(buf257, (1, 512, 1024), (524288, 1024, 1), 0); del buf257  # reuse
    # Source Nodes: [attn_out_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf258, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf259, (1, 1024, 1024), (0, 1024, 1), 0), out=buf260)
    buf261 = buf242; del buf242  # reuse
    buf262 = buf241; del buf241  # reuse
    buf264 = reinterpret_tensor(buf258, (512, 1, 1024), (1024, 1024, 1), 0); del buf258  # reuse
    cpp_fused_add_native_layer_norm_69(c_void_p(buf260.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()))
    del arg241_1
    del arg242_1
    buf265 = reinterpret_tensor(buf239, (512, 4096), (4096, 1), 0); del buf239  # reuse
    # Source Nodes: [output_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg244_1, reinterpret_tensor(buf264, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg243_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf265)
    del arg243_1
    del arg244_1
    buf266 = reinterpret_tensor(buf265, (512, 1, 4096), (4096, 4096, 1), 0); del buf265  # reuse
    cpp_fused_gelu_70(c_void_p(buf266.data_ptr()))
    buf267 = reinterpret_tensor(buf260, (512, 1024), (1024, 1), 0); del buf260  # reuse
    # Source Nodes: [output_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg246_1, reinterpret_tensor(buf266, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg245_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf267)
    del arg245_1
    del arg246_1
    buf268 = buf262; del buf262  # reuse
    buf269 = buf261; del buf261  # reuse
    buf271 = reinterpret_tensor(buf246, (512, 1, 1024), (1024, 1024, 1), 0); del buf246  # reuse
    cpp_fused_add_native_layer_norm_71(c_void_p(buf267.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf271.data_ptr()))
    del arg247_1
    del arg248_1
    buf272 = reinterpret_tensor(buf267, (1, 512, 1024), (524288, 1024, 1), 0); del buf267  # reuse
    # Source Nodes: [q_head_h_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf271, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg70_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf272)
    del arg70_1
    buf273 = reinterpret_tensor(buf264, (1, 512, 1024), (524288, 1024, 1), 0); del buf264  # reuse
    # Source Nodes: [k_head_h_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf271, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg71_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf273)
    del arg71_1
    buf274 = reinterpret_tensor(buf245, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf245  # reuse
    buf277 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_72(c_void_p(buf272.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf277.data_ptr()))
    del arg74_1
    del arg75_1
    buf275 = reinterpret_tensor(buf256, (16, 512, 512), (262144, 512, 1), 0); del buf256  # reuse
    # Source Nodes: [ac_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf274, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf273, (16, 64, 512), (64, 1, 1024), 0), out=buf275)
    buf276 = reinterpret_tensor(buf259, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf259  # reuse
    # Source Nodes: [k_head_r_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg73_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf276)
    del arg73_1
    buf278 = buf251; del buf251  # reuse
    # Source Nodes: [bd_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf277, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf276, (16, 64, 1024), (64, 1, 1024), 0), out=buf278)
    buf279 = buf254; del buf254  # reuse
    buf280 = reinterpret_tensor(buf275, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf275  # reuse
    buf281 = buf252; del buf252  # reuse
    cpp_fused__softmax_add_index_select_mul_73(c_void_p(buf280.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()))
    buf282 = reinterpret_tensor(buf277, (1, 512, 1024), (524288, 1024, 1), 0); del buf277  # reuse
    # Source Nodes: [v_head_h_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf271, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg72_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf282)
    del arg72_1
    buf283 = buf280; del buf280  # reuse
    cpp_fused__softmax_74(c_void_p(buf283.data_ptr()), c_void_p(buf281.data_ptr()))
    buf284 = reinterpret_tensor(buf274, (16, 512, 64), (32768, 64, 1), 0); del buf274  # reuse
    # Source Nodes: [attn_vec_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf283, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf282, (16, 512, 64), (64, 1024, 1), 0), out=buf284)
    buf285 = reinterpret_tensor(buf282, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf282  # reuse
    buf286 = reinterpret_tensor(buf276, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf276  # reuse
    cpp_fused_clone_75(c_void_p(buf284.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    del arg76_1
    buf287 = reinterpret_tensor(buf284, (1, 512, 1024), (524288, 1024, 1), 0); del buf284  # reuse
    # Source Nodes: [attn_out_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf285, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf286, (1, 1024, 1024), (0, 1024, 1), 0), out=buf287)
    buf288 = buf269; del buf269  # reuse
    buf289 = buf268; del buf268  # reuse
    buf291 = reinterpret_tensor(buf285, (512, 1, 1024), (1024, 1024, 1), 0); del buf285  # reuse
    cpp_fused_add_native_layer_norm_76(c_void_p(buf287.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()))
    del arg249_1
    del arg250_1
    buf292 = reinterpret_tensor(buf266, (512, 4096), (4096, 1), 0); del buf266  # reuse
    # Source Nodes: [output_82], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg252_1, reinterpret_tensor(buf291, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg251_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf292)
    del arg251_1
    del arg252_1
    buf293 = reinterpret_tensor(buf292, (512, 1, 4096), (4096, 4096, 1), 0); del buf292  # reuse
    cpp_fused_gelu_77(c_void_p(buf293.data_ptr()))
    buf294 = reinterpret_tensor(buf287, (512, 1024), (1024, 1), 0); del buf287  # reuse
    # Source Nodes: [output_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg254_1, reinterpret_tensor(buf293, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg253_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf294)
    del arg253_1
    del arg254_1
    buf295 = buf289; del buf289  # reuse
    buf296 = buf288; del buf288  # reuse
    buf298 = reinterpret_tensor(buf273, (512, 1, 1024), (1024, 1024, 1), 0); del buf273  # reuse
    cpp_fused_add_native_layer_norm_78(c_void_p(buf294.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()))
    del arg255_1
    del arg256_1
    buf299 = reinterpret_tensor(buf294, (1, 512, 1024), (524288, 1024, 1), 0); del buf294  # reuse
    # Source Nodes: [q_head_h_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf298, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg77_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf299)
    del arg77_1
    buf300 = reinterpret_tensor(buf291, (1, 512, 1024), (524288, 1024, 1), 0); del buf291  # reuse
    # Source Nodes: [k_head_h_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf298, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg78_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf300)
    del arg78_1
    buf301 = reinterpret_tensor(buf272, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf272  # reuse
    buf304 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_79(c_void_p(buf299.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf304.data_ptr()))
    del arg81_1
    del arg82_1
    buf302 = reinterpret_tensor(buf283, (16, 512, 512), (262144, 512, 1), 0); del buf283  # reuse
    # Source Nodes: [ac_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf301, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf300, (16, 64, 512), (64, 1, 1024), 0), out=buf302)
    buf303 = reinterpret_tensor(buf286, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf286  # reuse
    # Source Nodes: [k_head_r_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg80_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf303)
    del arg80_1
    buf305 = buf278; del buf278  # reuse
    # Source Nodes: [bd_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf304, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf303, (16, 64, 1024), (64, 1, 1024), 0), out=buf305)
    buf306 = buf281; del buf281  # reuse
    buf307 = reinterpret_tensor(buf302, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf302  # reuse
    buf308 = buf279; del buf279  # reuse
    cpp_fused__softmax_add_index_select_mul_80(c_void_p(buf307.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf308.data_ptr()))
    buf309 = reinterpret_tensor(buf304, (1, 512, 1024), (524288, 1024, 1), 0); del buf304  # reuse
    # Source Nodes: [v_head_h_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf298, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg79_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf309)
    del arg79_1
    buf310 = buf307; del buf307  # reuse
    cpp_fused__softmax_81(c_void_p(buf310.data_ptr()), c_void_p(buf308.data_ptr()))
    buf311 = reinterpret_tensor(buf301, (16, 512, 64), (32768, 64, 1), 0); del buf301  # reuse
    # Source Nodes: [attn_vec_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf310, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf309, (16, 512, 64), (64, 1024, 1), 0), out=buf311)
    buf312 = reinterpret_tensor(buf309, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf309  # reuse
    buf313 = reinterpret_tensor(buf303, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf303  # reuse
    cpp_fused_clone_82(c_void_p(buf311.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()))
    del arg83_1
    buf314 = reinterpret_tensor(buf311, (1, 512, 1024), (524288, 1024, 1), 0); del buf311  # reuse
    # Source Nodes: [attn_out_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf312, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf313, (1, 1024, 1024), (0, 1024, 1), 0), out=buf314)
    buf315 = buf296; del buf296  # reuse
    buf316 = buf295; del buf295  # reuse
    buf318 = reinterpret_tensor(buf312, (512, 1, 1024), (1024, 1024, 1), 0); del buf312  # reuse
    cpp_fused_add_native_layer_norm_83(c_void_p(buf314.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()))
    del arg257_1
    del arg258_1
    buf319 = reinterpret_tensor(buf293, (512, 4096), (4096, 1), 0); del buf293  # reuse
    # Source Nodes: [output_90], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg260_1, reinterpret_tensor(buf318, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg259_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf319)
    del arg259_1
    del arg260_1
    buf320 = reinterpret_tensor(buf319, (512, 1, 4096), (4096, 4096, 1), 0); del buf319  # reuse
    cpp_fused_gelu_84(c_void_p(buf320.data_ptr()))
    buf321 = reinterpret_tensor(buf314, (512, 1024), (1024, 1), 0); del buf314  # reuse
    # Source Nodes: [output_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg262_1, reinterpret_tensor(buf320, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg261_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf321)
    del arg261_1
    del arg262_1
    buf322 = buf316; del buf316  # reuse
    buf323 = buf315; del buf315  # reuse
    buf325 = reinterpret_tensor(buf300, (512, 1, 1024), (1024, 1024, 1), 0); del buf300  # reuse
    cpp_fused_add_native_layer_norm_85(c_void_p(buf321.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf325.data_ptr()))
    del arg263_1
    del arg264_1
    buf326 = reinterpret_tensor(buf321, (1, 512, 1024), (524288, 1024, 1), 0); del buf321  # reuse
    # Source Nodes: [q_head_h_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf325, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg84_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf326)
    del arg84_1
    buf327 = reinterpret_tensor(buf318, (1, 512, 1024), (524288, 1024, 1), 0); del buf318  # reuse
    # Source Nodes: [k_head_h_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf325, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg85_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf327)
    del arg85_1
    buf328 = reinterpret_tensor(buf299, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf299  # reuse
    buf331 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_86(c_void_p(buf326.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf331.data_ptr()))
    del arg88_1
    del arg89_1
    buf329 = reinterpret_tensor(buf310, (16, 512, 512), (262144, 512, 1), 0); del buf310  # reuse
    # Source Nodes: [ac_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf328, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf327, (16, 64, 512), (64, 1, 1024), 0), out=buf329)
    buf330 = reinterpret_tensor(buf313, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf313  # reuse
    # Source Nodes: [k_head_r_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg87_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf330)
    del arg87_1
    buf332 = buf305; del buf305  # reuse
    # Source Nodes: [bd_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf331, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf330, (16, 64, 1024), (64, 1, 1024), 0), out=buf332)
    buf333 = buf308; del buf308  # reuse
    buf334 = reinterpret_tensor(buf329, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf329  # reuse
    buf335 = buf306; del buf306  # reuse
    cpp_fused__softmax_add_index_select_mul_87(c_void_p(buf334.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()))
    buf336 = reinterpret_tensor(buf331, (1, 512, 1024), (524288, 1024, 1), 0); del buf331  # reuse
    # Source Nodes: [v_head_h_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf325, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg86_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf336)
    del arg86_1
    buf337 = buf334; del buf334  # reuse
    cpp_fused__softmax_88(c_void_p(buf337.data_ptr()), c_void_p(buf335.data_ptr()))
    buf338 = reinterpret_tensor(buf328, (16, 512, 64), (32768, 64, 1), 0); del buf328  # reuse
    # Source Nodes: [attn_vec_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf337, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf336, (16, 512, 64), (64, 1024, 1), 0), out=buf338)
    buf339 = reinterpret_tensor(buf336, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf336  # reuse
    buf340 = reinterpret_tensor(buf330, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf330  # reuse
    cpp_fused_clone_89(c_void_p(buf338.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    del arg90_1
    buf341 = reinterpret_tensor(buf338, (1, 512, 1024), (524288, 1024, 1), 0); del buf338  # reuse
    # Source Nodes: [attn_out_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf339, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf340, (1, 1024, 1024), (0, 1024, 1), 0), out=buf341)
    buf342 = buf323; del buf323  # reuse
    buf343 = buf322; del buf322  # reuse
    buf345 = reinterpret_tensor(buf339, (512, 1, 1024), (1024, 1024, 1), 0); del buf339  # reuse
    cpp_fused_add_native_layer_norm_90(c_void_p(buf341.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf345.data_ptr()))
    del arg265_1
    del arg266_1
    buf346 = reinterpret_tensor(buf320, (512, 4096), (4096, 1), 0); del buf320  # reuse
    # Source Nodes: [output_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg268_1, reinterpret_tensor(buf345, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg267_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf346)
    del arg267_1
    del arg268_1
    buf347 = reinterpret_tensor(buf346, (512, 1, 4096), (4096, 4096, 1), 0); del buf346  # reuse
    cpp_fused_gelu_91(c_void_p(buf347.data_ptr()))
    buf348 = reinterpret_tensor(buf341, (512, 1024), (1024, 1), 0); del buf341  # reuse
    # Source Nodes: [output_101], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg270_1, reinterpret_tensor(buf347, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg269_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf348)
    del arg269_1
    del arg270_1
    buf349 = buf343; del buf343  # reuse
    buf350 = buf342; del buf342  # reuse
    buf352 = reinterpret_tensor(buf327, (512, 1, 1024), (1024, 1024, 1), 0); del buf327  # reuse
    cpp_fused_add_native_layer_norm_92(c_void_p(buf348.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf352.data_ptr()))
    del arg271_1
    del arg272_1
    buf353 = reinterpret_tensor(buf348, (1, 512, 1024), (524288, 1024, 1), 0); del buf348  # reuse
    # Source Nodes: [q_head_h_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf352, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg91_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf353)
    del arg91_1
    buf354 = reinterpret_tensor(buf345, (1, 512, 1024), (524288, 1024, 1), 0); del buf345  # reuse
    # Source Nodes: [k_head_h_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf352, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg92_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf354)
    del arg92_1
    buf355 = reinterpret_tensor(buf326, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf326  # reuse
    buf358 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_93(c_void_p(buf353.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf358.data_ptr()))
    del arg95_1
    del arg96_1
    buf356 = reinterpret_tensor(buf337, (16, 512, 512), (262144, 512, 1), 0); del buf337  # reuse
    # Source Nodes: [ac_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf355, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf354, (16, 64, 512), (64, 1, 1024), 0), out=buf356)
    buf357 = reinterpret_tensor(buf340, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf340  # reuse
    # Source Nodes: [k_head_r_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg94_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf357)
    del arg94_1
    buf359 = buf332; del buf332  # reuse
    # Source Nodes: [bd_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf358, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf357, (16, 64, 1024), (64, 1, 1024), 0), out=buf359)
    buf360 = buf335; del buf335  # reuse
    buf361 = reinterpret_tensor(buf356, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf356  # reuse
    buf362 = buf333; del buf333  # reuse
    cpp_fused__softmax_add_index_select_mul_94(c_void_p(buf361.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf362.data_ptr()))
    buf363 = reinterpret_tensor(buf358, (1, 512, 1024), (524288, 1024, 1), 0); del buf358  # reuse
    # Source Nodes: [v_head_h_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf352, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg93_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf363)
    del arg93_1
    buf364 = buf361; del buf361  # reuse
    cpp_fused__softmax_95(c_void_p(buf364.data_ptr()), c_void_p(buf362.data_ptr()))
    buf365 = reinterpret_tensor(buf355, (16, 512, 64), (32768, 64, 1), 0); del buf355  # reuse
    # Source Nodes: [attn_vec_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf364, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf363, (16, 512, 64), (64, 1024, 1), 0), out=buf365)
    buf366 = reinterpret_tensor(buf363, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf363  # reuse
    buf367 = reinterpret_tensor(buf357, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf357  # reuse
    cpp_fused_clone_96(c_void_p(buf365.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()))
    del arg97_1
    buf368 = reinterpret_tensor(buf365, (1, 512, 1024), (524288, 1024, 1), 0); del buf365  # reuse
    # Source Nodes: [attn_out_39], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf366, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf367, (1, 1024, 1024), (0, 1024, 1), 0), out=buf368)
    buf369 = buf350; del buf350  # reuse
    buf370 = buf349; del buf349  # reuse
    buf372 = reinterpret_tensor(buf366, (512, 1, 1024), (1024, 1024, 1), 0); del buf366  # reuse
    cpp_fused_add_native_layer_norm_97(c_void_p(buf368.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf372.data_ptr()))
    del arg273_1
    del arg274_1
    buf373 = reinterpret_tensor(buf347, (512, 4096), (4096, 1), 0); del buf347  # reuse
    # Source Nodes: [output_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg276_1, reinterpret_tensor(buf372, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg275_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf373)
    del arg275_1
    del arg276_1
    buf374 = reinterpret_tensor(buf373, (512, 1, 4096), (4096, 4096, 1), 0); del buf373  # reuse
    cpp_fused_gelu_98(c_void_p(buf374.data_ptr()))
    buf375 = reinterpret_tensor(buf368, (512, 1024), (1024, 1), 0); del buf368  # reuse
    # Source Nodes: [output_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg278_1, reinterpret_tensor(buf374, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg277_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf375)
    del arg277_1
    del arg278_1
    buf376 = buf370; del buf370  # reuse
    buf377 = buf369; del buf369  # reuse
    buf379 = reinterpret_tensor(buf354, (512, 1, 1024), (1024, 1024, 1), 0); del buf354  # reuse
    cpp_fused_add_native_layer_norm_99(c_void_p(buf375.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf379.data_ptr()))
    del arg279_1
    del arg280_1
    buf380 = reinterpret_tensor(buf375, (1, 512, 1024), (524288, 1024, 1), 0); del buf375  # reuse
    # Source Nodes: [q_head_h_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf379, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg98_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf380)
    del arg98_1
    buf381 = reinterpret_tensor(buf372, (1, 512, 1024), (524288, 1024, 1), 0); del buf372  # reuse
    # Source Nodes: [k_head_h_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf379, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg99_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf381)
    del arg99_1
    buf382 = reinterpret_tensor(buf353, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf353  # reuse
    buf385 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_100(c_void_p(buf380.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf385.data_ptr()))
    del arg102_1
    del arg103_1
    buf383 = reinterpret_tensor(buf364, (16, 512, 512), (262144, 512, 1), 0); del buf364  # reuse
    # Source Nodes: [ac_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf382, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf381, (16, 64, 512), (64, 1, 1024), 0), out=buf383)
    buf384 = reinterpret_tensor(buf367, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf367  # reuse
    # Source Nodes: [k_head_r_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg101_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf384)
    del arg101_1
    buf386 = buf359; del buf359  # reuse
    # Source Nodes: [bd_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf385, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf384, (16, 64, 1024), (64, 1, 1024), 0), out=buf386)
    buf387 = buf362; del buf362  # reuse
    buf388 = reinterpret_tensor(buf383, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf383  # reuse
    buf389 = buf360; del buf360  # reuse
    cpp_fused__softmax_add_index_select_mul_101(c_void_p(buf388.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf389.data_ptr()))
    buf390 = reinterpret_tensor(buf385, (1, 512, 1024), (524288, 1024, 1), 0); del buf385  # reuse
    # Source Nodes: [v_head_h_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf379, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg100_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf390)
    del arg100_1
    buf391 = buf388; del buf388  # reuse
    cpp_fused__softmax_102(c_void_p(buf391.data_ptr()), c_void_p(buf389.data_ptr()))
    buf392 = reinterpret_tensor(buf382, (16, 512, 64), (32768, 64, 1), 0); del buf382  # reuse
    # Source Nodes: [attn_vec_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf391, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf390, (16, 512, 64), (64, 1024, 1), 0), out=buf392)
    buf393 = reinterpret_tensor(buf390, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf390  # reuse
    buf394 = reinterpret_tensor(buf384, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf384  # reuse
    cpp_fused_clone_103(c_void_p(buf392.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()))
    del arg104_1
    buf395 = reinterpret_tensor(buf392, (1, 512, 1024), (524288, 1024, 1), 0); del buf392  # reuse
    # Source Nodes: [attn_out_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf393, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf394, (1, 1024, 1024), (0, 1024, 1), 0), out=buf395)
    buf396 = buf377; del buf377  # reuse
    buf397 = buf376; del buf376  # reuse
    buf399 = reinterpret_tensor(buf393, (512, 1, 1024), (1024, 1024, 1), 0); del buf393  # reuse
    cpp_fused_add_native_layer_norm_104(c_void_p(buf395.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf399.data_ptr()))
    del arg281_1
    del arg282_1
    buf400 = reinterpret_tensor(buf374, (512, 4096), (4096, 1), 0); del buf374  # reuse
    # Source Nodes: [output_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg284_1, reinterpret_tensor(buf399, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg283_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf400)
    del arg283_1
    del arg284_1
    buf401 = reinterpret_tensor(buf400, (512, 1, 4096), (4096, 4096, 1), 0); del buf400  # reuse
    cpp_fused_gelu_105(c_void_p(buf401.data_ptr()))
    buf402 = reinterpret_tensor(buf395, (512, 1024), (1024, 1), 0); del buf395  # reuse
    # Source Nodes: [output_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg286_1, reinterpret_tensor(buf401, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg285_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf402)
    del arg285_1
    del arg286_1
    buf403 = buf397; del buf397  # reuse
    buf404 = buf396; del buf396  # reuse
    buf406 = reinterpret_tensor(buf381, (512, 1, 1024), (1024, 1024, 1), 0); del buf381  # reuse
    cpp_fused_add_native_layer_norm_106(c_void_p(buf402.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf406.data_ptr()))
    del arg287_1
    del arg288_1
    buf407 = reinterpret_tensor(buf402, (1, 512, 1024), (524288, 1024, 1), 0); del buf402  # reuse
    # Source Nodes: [q_head_h_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf406, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg105_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf407)
    del arg105_1
    buf408 = reinterpret_tensor(buf399, (1, 512, 1024), (524288, 1024, 1), 0); del buf399  # reuse
    # Source Nodes: [k_head_h_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf406, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg106_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf408)
    del arg106_1
    buf409 = reinterpret_tensor(buf380, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf380  # reuse
    buf412 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_107(c_void_p(buf407.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf412.data_ptr()))
    del arg109_1
    del arg110_1
    buf410 = reinterpret_tensor(buf391, (16, 512, 512), (262144, 512, 1), 0); del buf391  # reuse
    # Source Nodes: [ac_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf409, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf408, (16, 64, 512), (64, 1, 1024), 0), out=buf410)
    buf411 = reinterpret_tensor(buf394, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf394  # reuse
    # Source Nodes: [k_head_r_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg108_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf411)
    del arg108_1
    buf413 = buf386; del buf386  # reuse
    # Source Nodes: [bd_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf412, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf411, (16, 64, 1024), (64, 1, 1024), 0), out=buf413)
    buf414 = buf389; del buf389  # reuse
    buf415 = reinterpret_tensor(buf410, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf410  # reuse
    buf416 = buf387; del buf387  # reuse
    cpp_fused__softmax_add_index_select_mul_108(c_void_p(buf415.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf416.data_ptr()))
    buf417 = reinterpret_tensor(buf412, (1, 512, 1024), (524288, 1024, 1), 0); del buf412  # reuse
    # Source Nodes: [v_head_h_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf406, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg107_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf417)
    del arg107_1
    buf418 = buf415; del buf415  # reuse
    cpp_fused__softmax_109(c_void_p(buf418.data_ptr()), c_void_p(buf416.data_ptr()))
    buf419 = reinterpret_tensor(buf409, (16, 512, 64), (32768, 64, 1), 0); del buf409  # reuse
    # Source Nodes: [attn_vec_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf418, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf417, (16, 512, 64), (64, 1024, 1), 0), out=buf419)
    buf420 = reinterpret_tensor(buf417, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf417  # reuse
    buf421 = reinterpret_tensor(buf411, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf411  # reuse
    cpp_fused_clone_110(c_void_p(buf419.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()))
    del arg111_1
    buf422 = reinterpret_tensor(buf419, (1, 512, 1024), (524288, 1024, 1), 0); del buf419  # reuse
    # Source Nodes: [attn_out_45], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf420, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf421, (1, 1024, 1024), (0, 1024, 1), 0), out=buf422)
    buf423 = buf404; del buf404  # reuse
    buf424 = buf403; del buf403  # reuse
    buf426 = reinterpret_tensor(buf420, (512, 1, 1024), (1024, 1024, 1), 0); del buf420  # reuse
    cpp_fused_add_native_layer_norm_111(c_void_p(buf422.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf426.data_ptr()))
    del arg289_1
    del arg290_1
    buf427 = reinterpret_tensor(buf401, (512, 4096), (4096, 1), 0); del buf401  # reuse
    # Source Nodes: [output_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg292_1, reinterpret_tensor(buf426, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg291_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf427)
    del arg291_1
    del arg292_1
    buf428 = reinterpret_tensor(buf427, (512, 1, 4096), (4096, 4096, 1), 0); del buf427  # reuse
    cpp_fused_gelu_112(c_void_p(buf428.data_ptr()))
    buf429 = reinterpret_tensor(buf422, (512, 1024), (1024, 1), 0); del buf422  # reuse
    # Source Nodes: [output_125], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg294_1, reinterpret_tensor(buf428, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg293_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf429)
    del arg293_1
    del arg294_1
    buf430 = buf424; del buf424  # reuse
    buf431 = buf423; del buf423  # reuse
    buf433 = reinterpret_tensor(buf408, (512, 1, 1024), (1024, 1024, 1), 0); del buf408  # reuse
    cpp_fused_add_native_layer_norm_113(c_void_p(buf429.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf433.data_ptr()))
    del arg295_1
    del arg296_1
    buf434 = reinterpret_tensor(buf429, (1, 512, 1024), (524288, 1024, 1), 0); del buf429  # reuse
    # Source Nodes: [q_head_h_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf433, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg112_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf434)
    del arg112_1
    buf435 = reinterpret_tensor(buf426, (1, 512, 1024), (524288, 1024, 1), 0); del buf426  # reuse
    # Source Nodes: [k_head_h_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf433, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg113_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf435)
    del arg113_1
    buf436 = reinterpret_tensor(buf407, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf407  # reuse
    buf439 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_114(c_void_p(buf434.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf439.data_ptr()))
    del arg116_1
    del arg117_1
    buf437 = reinterpret_tensor(buf418, (16, 512, 512), (262144, 512, 1), 0); del buf418  # reuse
    # Source Nodes: [ac_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf436, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf435, (16, 64, 512), (64, 1, 1024), 0), out=buf437)
    buf438 = reinterpret_tensor(buf421, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf421  # reuse
    # Source Nodes: [k_head_r_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg115_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf438)
    del arg115_1
    buf440 = buf413; del buf413  # reuse
    # Source Nodes: [bd_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf439, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf438, (16, 64, 1024), (64, 1, 1024), 0), out=buf440)
    buf441 = buf416; del buf416  # reuse
    buf442 = reinterpret_tensor(buf437, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf437  # reuse
    buf443 = buf414; del buf414  # reuse
    cpp_fused__softmax_add_index_select_mul_115(c_void_p(buf442.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf443.data_ptr()))
    buf444 = reinterpret_tensor(buf439, (1, 512, 1024), (524288, 1024, 1), 0); del buf439  # reuse
    # Source Nodes: [v_head_h_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf433, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg114_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf444)
    del arg114_1
    buf445 = buf442; del buf442  # reuse
    cpp_fused__softmax_116(c_void_p(buf445.data_ptr()), c_void_p(buf443.data_ptr()))
    buf446 = reinterpret_tensor(buf436, (16, 512, 64), (32768, 64, 1), 0); del buf436  # reuse
    # Source Nodes: [attn_vec_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf445, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf444, (16, 512, 64), (64, 1024, 1), 0), out=buf446)
    buf447 = reinterpret_tensor(buf444, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf444  # reuse
    buf448 = reinterpret_tensor(buf438, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf438  # reuse
    cpp_fused_clone_117(c_void_p(buf446.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()))
    del arg118_1
    buf449 = reinterpret_tensor(buf446, (1, 512, 1024), (524288, 1024, 1), 0); del buf446  # reuse
    # Source Nodes: [attn_out_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf447, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf448, (1, 1024, 1024), (0, 1024, 1), 0), out=buf449)
    buf450 = buf431; del buf431  # reuse
    buf451 = buf430; del buf430  # reuse
    buf453 = reinterpret_tensor(buf447, (512, 1, 1024), (1024, 1024, 1), 0); del buf447  # reuse
    cpp_fused_add_native_layer_norm_118(c_void_p(buf449.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf453.data_ptr()))
    del arg297_1
    del arg298_1
    buf454 = reinterpret_tensor(buf428, (512, 4096), (4096, 1), 0); del buf428  # reuse
    # Source Nodes: [output_130], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg300_1, reinterpret_tensor(buf453, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg299_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf454)
    del arg299_1
    del arg300_1
    buf455 = reinterpret_tensor(buf454, (512, 1, 4096), (4096, 4096, 1), 0); del buf454  # reuse
    cpp_fused_gelu_119(c_void_p(buf455.data_ptr()))
    buf456 = reinterpret_tensor(buf449, (512, 1024), (1024, 1), 0); del buf449  # reuse
    # Source Nodes: [output_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg302_1, reinterpret_tensor(buf455, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg301_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf456)
    del arg301_1
    del arg302_1
    buf457 = buf451; del buf451  # reuse
    buf458 = buf450; del buf450  # reuse
    buf460 = reinterpret_tensor(buf435, (512, 1, 1024), (1024, 1024, 1), 0); del buf435  # reuse
    cpp_fused_add_native_layer_norm_120(c_void_p(buf456.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf460.data_ptr()))
    del arg303_1
    del arg304_1
    buf461 = reinterpret_tensor(buf456, (1, 512, 1024), (524288, 1024, 1), 0); del buf456  # reuse
    # Source Nodes: [q_head_h_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf460, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg119_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf461)
    del arg119_1
    buf462 = reinterpret_tensor(buf453, (1, 512, 1024), (524288, 1024, 1), 0); del buf453  # reuse
    # Source Nodes: [k_head_h_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf460, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg120_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf462)
    del arg120_1
    buf463 = reinterpret_tensor(buf434, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf434  # reuse
    buf466 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_121(c_void_p(buf461.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf466.data_ptr()))
    del arg123_1
    del arg124_1
    buf464 = reinterpret_tensor(buf445, (16, 512, 512), (262144, 512, 1), 0); del buf445  # reuse
    # Source Nodes: [ac_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf463, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf462, (16, 64, 512), (64, 1, 1024), 0), out=buf464)
    buf465 = reinterpret_tensor(buf448, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf448  # reuse
    # Source Nodes: [k_head_r_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg122_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf465)
    del arg122_1
    buf467 = buf440; del buf440  # reuse
    # Source Nodes: [bd_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf466, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf465, (16, 64, 1024), (64, 1, 1024), 0), out=buf467)
    buf468 = buf443; del buf443  # reuse
    buf469 = reinterpret_tensor(buf464, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf464  # reuse
    buf470 = buf441; del buf441  # reuse
    cpp_fused__softmax_add_index_select_mul_122(c_void_p(buf469.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf470.data_ptr()))
    buf471 = reinterpret_tensor(buf466, (1, 512, 1024), (524288, 1024, 1), 0); del buf466  # reuse
    # Source Nodes: [v_head_h_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf460, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg121_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf471)
    del arg121_1
    buf472 = buf469; del buf469  # reuse
    cpp_fused__softmax_123(c_void_p(buf472.data_ptr()), c_void_p(buf470.data_ptr()))
    buf473 = reinterpret_tensor(buf463, (16, 512, 64), (32768, 64, 1), 0); del buf463  # reuse
    # Source Nodes: [attn_vec_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf472, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf471, (16, 512, 64), (64, 1024, 1), 0), out=buf473)
    buf474 = reinterpret_tensor(buf471, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf471  # reuse
    buf475 = reinterpret_tensor(buf465, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf465  # reuse
    cpp_fused_clone_124(c_void_p(buf473.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()))
    del arg125_1
    buf476 = reinterpret_tensor(buf473, (1, 512, 1024), (524288, 1024, 1), 0); del buf473  # reuse
    # Source Nodes: [attn_out_51], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf474, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf475, (1, 1024, 1024), (0, 1024, 1), 0), out=buf476)
    buf477 = buf458; del buf458  # reuse
    buf478 = buf457; del buf457  # reuse
    buf480 = reinterpret_tensor(buf474, (512, 1, 1024), (1024, 1024, 1), 0); del buf474  # reuse
    cpp_fused_add_native_layer_norm_125(c_void_p(buf476.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf480.data_ptr()))
    del arg305_1
    del arg306_1
    buf481 = reinterpret_tensor(buf455, (512, 4096), (4096, 1), 0); del buf455  # reuse
    # Source Nodes: [output_138], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg308_1, reinterpret_tensor(buf480, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg307_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf481)
    del arg307_1
    del arg308_1
    buf482 = reinterpret_tensor(buf481, (512, 1, 4096), (4096, 4096, 1), 0); del buf481  # reuse
    cpp_fused_gelu_126(c_void_p(buf482.data_ptr()))
    buf483 = reinterpret_tensor(buf476, (512, 1024), (1024, 1), 0); del buf476  # reuse
    # Source Nodes: [output_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg310_1, reinterpret_tensor(buf482, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg309_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf483)
    del arg309_1
    del arg310_1
    buf484 = buf478; del buf478  # reuse
    buf485 = buf477; del buf477  # reuse
    buf487 = reinterpret_tensor(buf462, (512, 1, 1024), (1024, 1024, 1), 0); del buf462  # reuse
    cpp_fused_add_native_layer_norm_127(c_void_p(buf483.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf487.data_ptr()))
    del arg311_1
    del arg312_1
    buf488 = reinterpret_tensor(buf483, (1, 512, 1024), (524288, 1024, 1), 0); del buf483  # reuse
    # Source Nodes: [q_head_h_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf487, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg126_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf488)
    del arg126_1
    buf489 = reinterpret_tensor(buf480, (1, 512, 1024), (524288, 1024, 1), 0); del buf480  # reuse
    # Source Nodes: [k_head_h_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf487, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg127_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf489)
    del arg127_1
    buf490 = reinterpret_tensor(buf461, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf461  # reuse
    buf493 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_128(c_void_p(buf488.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf493.data_ptr()))
    del arg130_1
    del arg131_1
    buf491 = reinterpret_tensor(buf472, (16, 512, 512), (262144, 512, 1), 0); del buf472  # reuse
    # Source Nodes: [ac_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf490, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf489, (16, 64, 512), (64, 1, 1024), 0), out=buf491)
    buf492 = reinterpret_tensor(buf475, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf475  # reuse
    # Source Nodes: [k_head_r_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg129_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf492)
    del arg129_1
    buf494 = buf467; del buf467  # reuse
    # Source Nodes: [bd_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf493, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf492, (16, 64, 1024), (64, 1, 1024), 0), out=buf494)
    buf495 = buf470; del buf470  # reuse
    buf496 = reinterpret_tensor(buf491, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf491  # reuse
    buf497 = buf468; del buf468  # reuse
    cpp_fused__softmax_add_index_select_mul_129(c_void_p(buf496.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf497.data_ptr()))
    buf498 = reinterpret_tensor(buf493, (1, 512, 1024), (524288, 1024, 1), 0); del buf493  # reuse
    # Source Nodes: [v_head_h_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf487, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg128_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf498)
    del arg128_1
    buf499 = buf496; del buf496  # reuse
    cpp_fused__softmax_130(c_void_p(buf499.data_ptr()), c_void_p(buf497.data_ptr()))
    buf500 = reinterpret_tensor(buf490, (16, 512, 64), (32768, 64, 1), 0); del buf490  # reuse
    # Source Nodes: [attn_vec_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf499, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf498, (16, 512, 64), (64, 1024, 1), 0), out=buf500)
    buf501 = reinterpret_tensor(buf498, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf498  # reuse
    buf502 = reinterpret_tensor(buf492, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf492  # reuse
    cpp_fused_clone_131(c_void_p(buf500.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()))
    del arg132_1
    buf503 = reinterpret_tensor(buf500, (1, 512, 1024), (524288, 1024, 1), 0); del buf500  # reuse
    # Source Nodes: [attn_out_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf501, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf502, (1, 1024, 1024), (0, 1024, 1), 0), out=buf503)
    buf504 = buf485; del buf485  # reuse
    buf505 = buf484; del buf484  # reuse
    buf507 = reinterpret_tensor(buf501, (512, 1, 1024), (1024, 1024, 1), 0); del buf501  # reuse
    cpp_fused_add_native_layer_norm_132(c_void_p(buf503.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf507.data_ptr()))
    del arg313_1
    del arg314_1
    buf508 = reinterpret_tensor(buf482, (512, 4096), (4096, 1), 0); del buf482  # reuse
    # Source Nodes: [output_146], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg316_1, reinterpret_tensor(buf507, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg315_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf508)
    del arg315_1
    del arg316_1
    buf509 = reinterpret_tensor(buf508, (512, 1, 4096), (4096, 4096, 1), 0); del buf508  # reuse
    cpp_fused_gelu_133(c_void_p(buf509.data_ptr()))
    buf510 = reinterpret_tensor(buf503, (512, 1024), (1024, 1), 0); del buf503  # reuse
    # Source Nodes: [output_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg318_1, reinterpret_tensor(buf509, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg317_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf510)
    del arg317_1
    del arg318_1
    buf511 = buf505; del buf505  # reuse
    buf512 = buf504; del buf504  # reuse
    buf514 = reinterpret_tensor(buf489, (512, 1, 1024), (1024, 1024, 1), 0); del buf489  # reuse
    cpp_fused_add_native_layer_norm_134(c_void_p(buf510.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf514.data_ptr()))
    del arg319_1
    del arg320_1
    buf515 = reinterpret_tensor(buf510, (1, 512, 1024), (524288, 1024, 1), 0); del buf510  # reuse
    # Source Nodes: [q_head_h_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf514, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg133_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf515)
    del arg133_1
    buf516 = reinterpret_tensor(buf507, (1, 512, 1024), (524288, 1024, 1), 0); del buf507  # reuse
    # Source Nodes: [k_head_h_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf514, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg134_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf516)
    del arg134_1
    buf517 = reinterpret_tensor(buf488, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf488  # reuse
    buf520 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_135(c_void_p(buf515.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf520.data_ptr()))
    del arg137_1
    del arg138_1
    buf518 = reinterpret_tensor(buf499, (16, 512, 512), (262144, 512, 1), 0); del buf499  # reuse
    # Source Nodes: [ac_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf517, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf516, (16, 64, 512), (64, 1, 1024), 0), out=buf518)
    buf519 = reinterpret_tensor(buf502, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf502  # reuse
    # Source Nodes: [k_head_r_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg136_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf519)
    del arg136_1
    buf521 = buf494; del buf494  # reuse
    # Source Nodes: [bd_38], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf520, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf519, (16, 64, 1024), (64, 1, 1024), 0), out=buf521)
    buf522 = buf497; del buf497  # reuse
    buf523 = reinterpret_tensor(buf518, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf518  # reuse
    buf524 = buf495; del buf495  # reuse
    cpp_fused__softmax_add_index_select_mul_136(c_void_p(buf523.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf524.data_ptr()))
    buf525 = reinterpret_tensor(buf520, (1, 512, 1024), (524288, 1024, 1), 0); del buf520  # reuse
    # Source Nodes: [v_head_h_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf514, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg135_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf525)
    del arg135_1
    buf526 = buf523; del buf523  # reuse
    cpp_fused__softmax_137(c_void_p(buf526.data_ptr()), c_void_p(buf524.data_ptr()))
    buf527 = reinterpret_tensor(buf517, (16, 512, 64), (32768, 64, 1), 0); del buf517  # reuse
    # Source Nodes: [attn_vec_38], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf526, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf525, (16, 512, 64), (64, 1024, 1), 0), out=buf527)
    buf528 = reinterpret_tensor(buf525, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf525  # reuse
    buf529 = reinterpret_tensor(buf519, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf519  # reuse
    cpp_fused_clone_138(c_void_p(buf527.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()))
    del arg139_1
    buf530 = reinterpret_tensor(buf527, (1, 512, 1024), (524288, 1024, 1), 0); del buf527  # reuse
    # Source Nodes: [attn_out_57], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf528, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf529, (1, 1024, 1024), (0, 1024, 1), 0), out=buf530)
    buf531 = buf512; del buf512  # reuse
    buf532 = buf511; del buf511  # reuse
    buf534 = reinterpret_tensor(buf528, (512, 1, 1024), (1024, 1024, 1), 0); del buf528  # reuse
    cpp_fused_add_native_layer_norm_139(c_void_p(buf530.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf534.data_ptr()))
    del arg321_1
    del arg322_1
    buf535 = reinterpret_tensor(buf509, (512, 4096), (4096, 1), 0); del buf509  # reuse
    # Source Nodes: [output_154], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg324_1, reinterpret_tensor(buf534, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg323_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf535)
    del arg323_1
    del arg324_1
    buf536 = reinterpret_tensor(buf535, (512, 1, 4096), (4096, 4096, 1), 0); del buf535  # reuse
    cpp_fused_gelu_140(c_void_p(buf536.data_ptr()))
    buf537 = reinterpret_tensor(buf530, (512, 1024), (1024, 1), 0); del buf530  # reuse
    # Source Nodes: [output_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg326_1, reinterpret_tensor(buf536, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg325_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf537)
    del arg325_1
    del arg326_1
    buf538 = buf532; del buf532  # reuse
    buf539 = buf531; del buf531  # reuse
    buf541 = reinterpret_tensor(buf516, (512, 1, 1024), (1024, 1024, 1), 0); del buf516  # reuse
    cpp_fused_add_native_layer_norm_141(c_void_p(buf537.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf541.data_ptr()))
    del arg327_1
    del arg328_1
    buf542 = reinterpret_tensor(buf537, (1, 512, 1024), (524288, 1024, 1), 0); del buf537  # reuse
    # Source Nodes: [q_head_h_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf541, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg140_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf542)
    del arg140_1
    buf543 = reinterpret_tensor(buf534, (1, 512, 1024), (524288, 1024, 1), 0); del buf534  # reuse
    # Source Nodes: [k_head_h_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf541, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg141_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf543)
    del arg141_1
    buf544 = reinterpret_tensor(buf515, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf515  # reuse
    buf547 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_142(c_void_p(buf542.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf547.data_ptr()))
    del arg144_1
    del arg145_1
    buf545 = reinterpret_tensor(buf526, (16, 512, 512), (262144, 512, 1), 0); del buf526  # reuse
    # Source Nodes: [ac_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf544, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf543, (16, 64, 512), (64, 1, 1024), 0), out=buf545)
    buf546 = reinterpret_tensor(buf529, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf529  # reuse
    # Source Nodes: [k_head_r_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg143_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf546)
    del arg143_1
    buf548 = buf521; del buf521  # reuse
    # Source Nodes: [bd_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf547, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf546, (16, 64, 1024), (64, 1, 1024), 0), out=buf548)
    buf549 = buf524; del buf524  # reuse
    buf550 = reinterpret_tensor(buf545, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf545  # reuse
    buf551 = buf522; del buf522  # reuse
    cpp_fused__softmax_add_index_select_mul_143(c_void_p(buf550.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf551.data_ptr()))
    buf552 = reinterpret_tensor(buf547, (1, 512, 1024), (524288, 1024, 1), 0); del buf547  # reuse
    # Source Nodes: [v_head_h_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf541, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg142_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf552)
    del arg142_1
    buf553 = buf550; del buf550  # reuse
    cpp_fused__softmax_144(c_void_p(buf553.data_ptr()), c_void_p(buf551.data_ptr()))
    buf554 = reinterpret_tensor(buf544, (16, 512, 64), (32768, 64, 1), 0); del buf544  # reuse
    # Source Nodes: [attn_vec_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf553, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf552, (16, 512, 64), (64, 1024, 1), 0), out=buf554)
    buf555 = reinterpret_tensor(buf552, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf552  # reuse
    buf556 = reinterpret_tensor(buf546, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf546  # reuse
    cpp_fused_clone_145(c_void_p(buf554.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf556.data_ptr()))
    del arg146_1
    buf557 = reinterpret_tensor(buf554, (1, 512, 1024), (524288, 1024, 1), 0); del buf554  # reuse
    # Source Nodes: [attn_out_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf555, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf556, (1, 1024, 1024), (0, 1024, 1), 0), out=buf557)
    buf558 = buf539; del buf539  # reuse
    buf559 = buf538; del buf538  # reuse
    buf561 = reinterpret_tensor(buf555, (512, 1, 1024), (1024, 1024, 1), 0); del buf555  # reuse
    cpp_fused_add_native_layer_norm_146(c_void_p(buf557.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf561.data_ptr()))
    del arg329_1
    del arg330_1
    buf562 = reinterpret_tensor(buf536, (512, 4096), (4096, 1), 0); del buf536  # reuse
    # Source Nodes: [output_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg332_1, reinterpret_tensor(buf561, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg331_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf562)
    del arg331_1
    del arg332_1
    buf563 = reinterpret_tensor(buf562, (512, 1, 4096), (4096, 4096, 1), 0); del buf562  # reuse
    cpp_fused_gelu_147(c_void_p(buf563.data_ptr()))
    buf564 = reinterpret_tensor(buf557, (512, 1024), (1024, 1), 0); del buf557  # reuse
    # Source Nodes: [output_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg334_1, reinterpret_tensor(buf563, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg333_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf564)
    del arg333_1
    del arg334_1
    buf565 = buf559; del buf559  # reuse
    buf566 = buf558; del buf558  # reuse
    buf568 = reinterpret_tensor(buf543, (512, 1, 1024), (1024, 1024, 1), 0); del buf543  # reuse
    cpp_fused_add_native_layer_norm_148(c_void_p(buf564.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf568.data_ptr()))
    del arg335_1
    del arg336_1
    buf569 = reinterpret_tensor(buf564, (1, 512, 1024), (524288, 1024, 1), 0); del buf564  # reuse
    # Source Nodes: [q_head_h_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf568, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg147_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf569)
    del arg147_1
    buf570 = reinterpret_tensor(buf561, (1, 512, 1024), (524288, 1024, 1), 0); del buf561  # reuse
    # Source Nodes: [k_head_h_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf568, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg148_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf570)
    del arg148_1
    buf571 = reinterpret_tensor(buf542, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf542  # reuse
    buf574 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_149(c_void_p(buf569.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf574.data_ptr()))
    del arg151_1
    del arg152_1
    buf572 = reinterpret_tensor(buf553, (16, 512, 512), (262144, 512, 1), 0); del buf553  # reuse
    # Source Nodes: [ac_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf571, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf570, (16, 64, 512), (64, 1, 1024), 0), out=buf572)
    buf573 = reinterpret_tensor(buf556, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf556  # reuse
    # Source Nodes: [k_head_r_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg150_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf573)
    del arg150_1
    buf575 = buf548; del buf548  # reuse
    # Source Nodes: [bd_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf574, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf573, (16, 64, 1024), (64, 1, 1024), 0), out=buf575)
    buf576 = buf551; del buf551  # reuse
    buf577 = reinterpret_tensor(buf572, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf572  # reuse
    buf578 = buf549; del buf549  # reuse
    cpp_fused__softmax_add_index_select_mul_150(c_void_p(buf577.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf578.data_ptr()))
    buf579 = reinterpret_tensor(buf574, (1, 512, 1024), (524288, 1024, 1), 0); del buf574  # reuse
    # Source Nodes: [v_head_h_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf568, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg149_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf579)
    del arg149_1
    buf580 = buf577; del buf577  # reuse
    cpp_fused__softmax_151(c_void_p(buf580.data_ptr()), c_void_p(buf578.data_ptr()))
    buf581 = reinterpret_tensor(buf571, (16, 512, 64), (32768, 64, 1), 0); del buf571  # reuse
    # Source Nodes: [attn_vec_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf580, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf579, (16, 512, 64), (64, 1024, 1), 0), out=buf581)
    buf582 = reinterpret_tensor(buf579, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf579  # reuse
    buf583 = reinterpret_tensor(buf573, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf573  # reuse
    cpp_fused_clone_152(c_void_p(buf581.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(buf583.data_ptr()))
    del arg153_1
    buf584 = reinterpret_tensor(buf581, (1, 512, 1024), (524288, 1024, 1), 0); del buf581  # reuse
    # Source Nodes: [attn_out_63], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf582, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf583, (1, 1024, 1024), (0, 1024, 1), 0), out=buf584)
    buf585 = buf566; del buf566  # reuse
    buf586 = buf565; del buf565  # reuse
    buf588 = reinterpret_tensor(buf582, (512, 1, 1024), (1024, 1024, 1), 0); del buf582  # reuse
    cpp_fused_add_native_layer_norm_153(c_void_p(buf584.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf588.data_ptr()))
    del arg337_1
    del arg338_1
    buf589 = reinterpret_tensor(buf563, (512, 4096), (4096, 1), 0); del buf563  # reuse
    # Source Nodes: [output_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg340_1, reinterpret_tensor(buf588, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg339_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf589)
    del arg339_1
    del arg340_1
    buf590 = reinterpret_tensor(buf589, (512, 1, 4096), (4096, 4096, 1), 0); del buf589  # reuse
    cpp_fused_gelu_154(c_void_p(buf590.data_ptr()))
    buf591 = reinterpret_tensor(buf584, (512, 1024), (1024, 1), 0); del buf584  # reuse
    # Source Nodes: [output_173], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg342_1, reinterpret_tensor(buf590, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg341_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf591)
    del arg341_1
    del arg342_1
    buf592 = buf586; del buf586  # reuse
    buf593 = buf585; del buf585  # reuse
    buf595 = reinterpret_tensor(buf570, (512, 1, 1024), (1024, 1024, 1), 0); del buf570  # reuse
    cpp_fused_add_native_layer_norm_155(c_void_p(buf591.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf595.data_ptr()))
    del arg343_1
    del arg344_1
    buf596 = reinterpret_tensor(buf591, (1, 512, 1024), (524288, 1024, 1), 0); del buf591  # reuse
    # Source Nodes: [q_head_h_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf595, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg154_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf596)
    del arg154_1
    buf597 = reinterpret_tensor(buf588, (1, 512, 1024), (524288, 1024, 1), 0); del buf588  # reuse
    # Source Nodes: [k_head_h_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf595, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg155_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf597)
    del arg155_1
    buf598 = reinterpret_tensor(buf569, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf569  # reuse
    buf601 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_156(c_void_p(buf596.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf601.data_ptr()))
    del arg158_1
    del arg159_1
    buf599 = reinterpret_tensor(buf580, (16, 512, 512), (262144, 512, 1), 0); del buf580  # reuse
    # Source Nodes: [ac_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf598, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf597, (16, 64, 512), (64, 1, 1024), 0), out=buf599)
    buf600 = reinterpret_tensor(buf583, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf583  # reuse
    # Source Nodes: [k_head_r_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg157_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf600)
    del arg157_1
    buf602 = buf575; del buf575  # reuse
    # Source Nodes: [bd_44], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf601, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf600, (16, 64, 1024), (64, 1, 1024), 0), out=buf602)
    buf603 = buf578; del buf578  # reuse
    buf604 = reinterpret_tensor(buf599, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf599  # reuse
    buf605 = buf576; del buf576  # reuse
    cpp_fused__softmax_add_index_select_mul_157(c_void_p(buf604.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf605.data_ptr()))
    buf606 = reinterpret_tensor(buf601, (1, 512, 1024), (524288, 1024, 1), 0); del buf601  # reuse
    # Source Nodes: [v_head_h_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf595, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg156_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf606)
    del arg156_1
    buf607 = buf604; del buf604  # reuse
    cpp_fused__softmax_158(c_void_p(buf607.data_ptr()), c_void_p(buf605.data_ptr()))
    buf608 = reinterpret_tensor(buf598, (16, 512, 64), (32768, 64, 1), 0); del buf598  # reuse
    # Source Nodes: [attn_vec_44], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf607, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf606, (16, 512, 64), (64, 1024, 1), 0), out=buf608)
    buf609 = reinterpret_tensor(buf606, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf606  # reuse
    buf610 = reinterpret_tensor(buf600, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf600  # reuse
    cpp_fused_clone_159(c_void_p(buf608.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf610.data_ptr()))
    del arg160_1
    buf611 = reinterpret_tensor(buf608, (1, 512, 1024), (524288, 1024, 1), 0); del buf608  # reuse
    # Source Nodes: [attn_out_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf609, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf610, (1, 1024, 1024), (0, 1024, 1), 0), out=buf611)
    buf612 = buf593; del buf593  # reuse
    buf613 = buf592; del buf592  # reuse
    buf615 = reinterpret_tensor(buf609, (512, 1, 1024), (1024, 1024, 1), 0); del buf609  # reuse
    cpp_fused_add_native_layer_norm_160(c_void_p(buf611.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf615.data_ptr()))
    del arg345_1
    del arg346_1
    buf616 = reinterpret_tensor(buf590, (512, 4096), (4096, 1), 0); del buf590  # reuse
    # Source Nodes: [output_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg348_1, reinterpret_tensor(buf615, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg347_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf616)
    del arg347_1
    del arg348_1
    buf617 = reinterpret_tensor(buf616, (512, 1, 4096), (4096, 4096, 1), 0); del buf616  # reuse
    cpp_fused_gelu_161(c_void_p(buf617.data_ptr()))
    buf618 = reinterpret_tensor(buf611, (512, 1024), (1024, 1), 0); del buf611  # reuse
    # Source Nodes: [output_181], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg350_1, reinterpret_tensor(buf617, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg349_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf618)
    del arg349_1
    del arg350_1
    buf619 = buf613; del buf613  # reuse
    buf620 = buf612; del buf612  # reuse
    buf622 = reinterpret_tensor(buf597, (512, 1, 1024), (1024, 1024, 1), 0); del buf597  # reuse
    cpp_fused_add_native_layer_norm_162(c_void_p(buf618.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf622.data_ptr()))
    del arg351_1
    del arg352_1
    buf623 = reinterpret_tensor(buf618, (1, 512, 1024), (524288, 1024, 1), 0); del buf618  # reuse
    # Source Nodes: [q_head_h_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf622, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg161_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf623)
    del arg161_1
    buf624 = reinterpret_tensor(buf615, (1, 512, 1024), (524288, 1024, 1), 0); del buf615  # reuse
    # Source Nodes: [k_head_h_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf622, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg162_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf624)
    del arg162_1
    buf625 = reinterpret_tensor(buf596, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf596  # reuse
    buf628 = empty((512, 1, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_163(c_void_p(buf623.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf628.data_ptr()))
    del arg165_1
    del arg166_1
    del buf623
    buf626 = reinterpret_tensor(buf607, (16, 512, 512), (262144, 512, 1), 0); del buf607  # reuse
    # Source Nodes: [ac_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf625, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf624, (16, 64, 512), (64, 1, 1024), 0), out=buf626)
    buf627 = reinterpret_tensor(buf610, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf610  # reuse
    # Source Nodes: [k_head_r_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg164_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf627)
    del arg164_1
    del buf5
    buf629 = buf602; del buf602  # reuse
    # Source Nodes: [bd_46], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf628, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf627, (16, 64, 1024), (64, 1, 1024), 0), out=buf629)
    buf630 = buf605; del buf605  # reuse
    buf631 = reinterpret_tensor(buf626, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf626  # reuse
    buf632 = buf603; del buf603  # reuse
    cpp_fused__softmax_add_index_select_mul_164(c_void_p(buf631.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf632.data_ptr()))
    del buf629
    del buf630
    buf633 = reinterpret_tensor(buf628, (1, 512, 1024), (524288, 1024, 1), 0); del buf628  # reuse
    # Source Nodes: [v_head_h_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf622, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg163_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf633)
    del arg163_1
    buf634 = buf631; del buf631  # reuse
    cpp_fused__softmax_165(c_void_p(buf634.data_ptr()), c_void_p(buf632.data_ptr()))
    del buf632
    buf635 = reinterpret_tensor(buf625, (16, 512, 64), (32768, 64, 1), 0); del buf625  # reuse
    # Source Nodes: [attn_vec_46], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf634, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf633, (16, 512, 64), (64, 1024, 1), 0), out=buf635)
    del buf634
    buf636 = reinterpret_tensor(buf633, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf633  # reuse
    buf637 = reinterpret_tensor(buf627, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf627  # reuse
    cpp_fused_clone_166(c_void_p(buf635.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(buf636.data_ptr()), c_void_p(buf637.data_ptr()))
    del arg167_1
    buf638 = reinterpret_tensor(buf635, (1, 512, 1024), (524288, 1024, 1), 0); del buf635  # reuse
    # Source Nodes: [attn_out_69], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf636, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf637, (1, 1024, 1024), (0, 1024, 1), 0), out=buf638)
    del buf637
    buf639 = buf620; del buf620  # reuse
    buf640 = buf619; del buf619  # reuse
    buf642 = reinterpret_tensor(buf636, (512, 1, 1024), (1024, 1024, 1), 0); del buf636  # reuse
    cpp_fused_add_native_layer_norm_167(c_void_p(buf638.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf642.data_ptr()))
    del arg353_1
    del arg354_1
    buf643 = reinterpret_tensor(buf617, (512, 4096), (4096, 1), 0); del buf617  # reuse
    # Source Nodes: [output_186], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg356_1, reinterpret_tensor(buf642, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg355_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf643)
    del arg355_1
    del arg356_1
    buf644 = reinterpret_tensor(buf643, (512, 1, 4096), (4096, 4096, 1), 0); del buf643  # reuse
    cpp_fused_gelu_168(c_void_p(buf644.data_ptr()))
    buf645 = reinterpret_tensor(buf638, (512, 1024), (1024, 1), 0); del buf638  # reuse
    # Source Nodes: [output_189], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg358_1, reinterpret_tensor(buf644, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg357_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf645)
    del arg357_1
    del arg358_1
    del buf644
    buf646 = buf640; del buf640  # reuse
    buf647 = buf639; del buf639  # reuse
    buf649 = reinterpret_tensor(buf624, (512, 1, 1024), (1024, 1024, 1), 0); del buf624  # reuse
    cpp_fused_add_native_layer_norm_169(c_void_p(buf645.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf649.data_ptr()))
    del arg359_1
    del arg360_1
    del buf642
    del buf645
    buf650 = empty((512, 32000), device='cpu', dtype=torch.float32)
    # Source Nodes: [logits], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg362_1, reinterpret_tensor(buf649, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg361_1, (1024, 32000), (1, 1024), 0), alpha=1, beta=1, out=buf650)
    del arg361_1
    del arg362_1
    del buf649
    buf651 = reinterpret_tensor(buf647, (512, 1), (1, 512), 0); del buf647  # reuse
    buf652 = reinterpret_tensor(buf646, (512, 1), (1, 512), 0); del buf646  # reuse
    buf653 = empty((), device='cpu', dtype=torch.float32)
    buf654 = empty((), device='cpu', dtype=torch.int64)
    buf655 = buf653; del buf653  # reuse
    cpp_fused__log_softmax_nll_loss_forward_170(c_void_p(buf655.data_ptr()), c_void_p(buf650.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf654.data_ptr()))
    del arg364_1
    return (buf655, reinterpret_tensor(buf650, (1, 512, 32000), (16384000, 32000, 1), 0), buf0, buf28, buf55, buf82, buf109, buf136, buf163, buf190, buf217, buf244, buf271, buf298, buf325, buf352, buf379, buf406, buf433, buf460, buf487, buf514, buf541, buf568, buf595, buf622, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((16, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((32000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((32000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((32000, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg364_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('XLNetLMHeadModel', benchmark_compiled_module)
