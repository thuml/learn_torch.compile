
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
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp11 = in_ptr4[static_cast<long>(x0)];
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
                        auto tmp12 = decltype(tmp11)(tmp11 + 2);
                        auto tmp13 = tmp11 < 0;
                        auto tmp14 = tmp13 ? tmp12 : tmp11;
                        TORCH_CHECK((0 <= tmp14) & (tmp14 < 2L), "index out of bounds: 0 <= tmp14 < 2L")
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*tmp14)));
                        auto tmp16 = tmp10 + tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp16);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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


cpp_fused_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (512L*x1)), static_cast<long>(512L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(6);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(12);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
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


cpp_fused_gelu_7 = async_compile.cpp('''
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


cpp_fused_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (512L*x1)), static_cast<long>(512L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(6);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(12);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
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


cpp_fused_gelu_15 = async_compile.cpp('''
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


cpp_fused_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (512L*x1)), static_cast<long>(512L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(6);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(12);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (512L*x1)), static_cast<long>(512L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(6);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(12);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
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


cpp_fused_gelu_31 = async_compile.cpp('''
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


cpp_fused_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (512L*x1)), static_cast<long>(512L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(6);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(12);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
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


cpp_fused_gelu_39 = async_compile.cpp('''
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


cpp_fused_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (512L*x1)), static_cast<long>(512L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(6);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(12);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
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


cpp_fused_gelu_47 = async_compile.cpp('''
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


cpp_fused_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (512L*x1)), static_cast<long>(512L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(6);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(12);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
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


cpp_fused_gelu_55 = async_compile.cpp('''
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


cpp_fused_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (512L*x1)), static_cast<long>(512L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(6);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(12);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
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


cpp_fused_gelu_63 = async_compile.cpp('''
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


cpp_fused_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (512L*x1)), static_cast<long>(512L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(6);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(12);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
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


cpp_fused_gelu_71 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_72 = async_compile.cpp('''
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


cpp_fused_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (512L*x1)), static_cast<long>(512L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(6);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(12);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
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


cpp_fused_gelu_79 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_80 = async_compile.cpp('''
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


cpp_fused_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (512L*x1)), static_cast<long>(512L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(6);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(12);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
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


cpp_fused_gelu_87 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_88 = async_compile.cpp('''
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


cpp_fused_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                tmp0.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (512L*x1)), static_cast<long>(512L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(6);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(12);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (64L*x1) + (384L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_94 = async_compile.cpp('''
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


cpp_fused_gelu_95 = async_compile.cpp('''
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


cpp_fused_gelu_native_layer_norm_97 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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


cpp_fused__log_softmax_nll_loss_forward_98 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1 = args
    args.clear()
    assert_size_stride(arg0_1, (384, 1), (1, 1))
    assert_size_stride(arg1_1, (384, 1), (1, 1))
    assert_size_stride(arg2_1, (384, 1), (1, 1))
    assert_size_stride(arg3_1, (384, 1), (1, 1))
    assert_size_stride(arg4_1, (384, 1), (1, 1))
    assert_size_stride(arg5_1, (384, 1), (1, 1))
    assert_size_stride(arg6_1, (384, 1), (1, 1))
    assert_size_stride(arg7_1, (384, 1), (1, 1))
    assert_size_stride(arg8_1, (384, 1), (1, 1))
    assert_size_stride(arg9_1, (384, 1), (1, 1))
    assert_size_stride(arg10_1, (384, 1), (1, 1))
    assert_size_stride(arg11_1, (384, 1), (1, 1))
    assert_size_stride(arg12_1, (30522, 768), (768, 1))
    assert_size_stride(arg13_1, (512, 768), (768, 1))
    assert_size_stride(arg14_1, (2, 768), (768, 1))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (384, 768), (768, 1))
    assert_size_stride(arg18_1, (384, ), (1, ))
    assert_size_stride(arg19_1, (384, 768), (768, 1))
    assert_size_stride(arg20_1, (384, ), (1, ))
    assert_size_stride(arg21_1, (384, 768), (768, 1))
    assert_size_stride(arg22_1, (384, ), (1, ))
    assert_size_stride(arg23_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg24_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg25_1, (54, 384), (384, 1))
    assert_size_stride(arg26_1, (54, ), (1, ))
    assert_size_stride(arg27_1, (384, 768), (768, 1))
    assert_size_stride(arg28_1, (384, ), (1, ))
    assert_size_stride(arg29_1, (768, 768), (768, 1))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (3072, 768), (768, 1))
    assert_size_stride(arg34_1, (3072, ), (1, ))
    assert_size_stride(arg35_1, (768, 3072), (3072, 1))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (384, 768), (768, 1))
    assert_size_stride(arg40_1, (384, ), (1, ))
    assert_size_stride(arg41_1, (384, 768), (768, 1))
    assert_size_stride(arg42_1, (384, ), (1, ))
    assert_size_stride(arg43_1, (384, 768), (768, 1))
    assert_size_stride(arg44_1, (384, ), (1, ))
    assert_size_stride(arg45_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg46_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg47_1, (54, 384), (384, 1))
    assert_size_stride(arg48_1, (54, ), (1, ))
    assert_size_stride(arg49_1, (384, 768), (768, 1))
    assert_size_stride(arg50_1, (384, ), (1, ))
    assert_size_stride(arg51_1, (768, 768), (768, 1))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (3072, 768), (768, 1))
    assert_size_stride(arg56_1, (3072, ), (1, ))
    assert_size_stride(arg57_1, (768, 3072), (3072, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (384, 768), (768, 1))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (384, 768), (768, 1))
    assert_size_stride(arg64_1, (384, ), (1, ))
    assert_size_stride(arg65_1, (384, 768), (768, 1))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg68_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg69_1, (54, 384), (384, 1))
    assert_size_stride(arg70_1, (54, ), (1, ))
    assert_size_stride(arg71_1, (384, 768), (768, 1))
    assert_size_stride(arg72_1, (384, ), (1, ))
    assert_size_stride(arg73_1, (768, 768), (768, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (3072, 768), (768, 1))
    assert_size_stride(arg78_1, (3072, ), (1, ))
    assert_size_stride(arg79_1, (768, 3072), (3072, 1))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (384, 768), (768, 1))
    assert_size_stride(arg84_1, (384, ), (1, ))
    assert_size_stride(arg85_1, (384, 768), (768, 1))
    assert_size_stride(arg86_1, (384, ), (1, ))
    assert_size_stride(arg87_1, (384, 768), (768, 1))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg90_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg91_1, (54, 384), (384, 1))
    assert_size_stride(arg92_1, (54, ), (1, ))
    assert_size_stride(arg93_1, (384, 768), (768, 1))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (768, 768), (768, 1))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (3072, 768), (768, 1))
    assert_size_stride(arg100_1, (3072, ), (1, ))
    assert_size_stride(arg101_1, (768, 3072), (3072, 1))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (384, 768), (768, 1))
    assert_size_stride(arg106_1, (384, ), (1, ))
    assert_size_stride(arg107_1, (384, 768), (768, 1))
    assert_size_stride(arg108_1, (384, ), (1, ))
    assert_size_stride(arg109_1, (384, 768), (768, 1))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg112_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg113_1, (54, 384), (384, 1))
    assert_size_stride(arg114_1, (54, ), (1, ))
    assert_size_stride(arg115_1, (384, 768), (768, 1))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (768, 768), (768, 1))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (3072, 768), (768, 1))
    assert_size_stride(arg122_1, (3072, ), (1, ))
    assert_size_stride(arg123_1, (768, 3072), (3072, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (384, 768), (768, 1))
    assert_size_stride(arg128_1, (384, ), (1, ))
    assert_size_stride(arg129_1, (384, 768), (768, 1))
    assert_size_stride(arg130_1, (384, ), (1, ))
    assert_size_stride(arg131_1, (384, 768), (768, 1))
    assert_size_stride(arg132_1, (384, ), (1, ))
    assert_size_stride(arg133_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg134_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg135_1, (54, 384), (384, 1))
    assert_size_stride(arg136_1, (54, ), (1, ))
    assert_size_stride(arg137_1, (384, 768), (768, 1))
    assert_size_stride(arg138_1, (384, ), (1, ))
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
    assert_size_stride(arg149_1, (384, 768), (768, 1))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (384, 768), (768, 1))
    assert_size_stride(arg152_1, (384, ), (1, ))
    assert_size_stride(arg153_1, (384, 768), (768, 1))
    assert_size_stride(arg154_1, (384, ), (1, ))
    assert_size_stride(arg155_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg156_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg157_1, (54, 384), (384, 1))
    assert_size_stride(arg158_1, (54, ), (1, ))
    assert_size_stride(arg159_1, (384, 768), (768, 1))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (768, 768), (768, 1))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (3072, 768), (768, 1))
    assert_size_stride(arg166_1, (3072, ), (1, ))
    assert_size_stride(arg167_1, (768, 3072), (3072, 1))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (384, 768), (768, 1))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (384, 768), (768, 1))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (384, 768), (768, 1))
    assert_size_stride(arg176_1, (384, ), (1, ))
    assert_size_stride(arg177_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg178_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg179_1, (54, 384), (384, 1))
    assert_size_stride(arg180_1, (54, ), (1, ))
    assert_size_stride(arg181_1, (384, 768), (768, 1))
    assert_size_stride(arg182_1, (384, ), (1, ))
    assert_size_stride(arg183_1, (768, 768), (768, 1))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (3072, 768), (768, 1))
    assert_size_stride(arg188_1, (3072, ), (1, ))
    assert_size_stride(arg189_1, (768, 3072), (3072, 1))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (768, ), (1, ))
    assert_size_stride(arg193_1, (384, 768), (768, 1))
    assert_size_stride(arg194_1, (384, ), (1, ))
    assert_size_stride(arg195_1, (384, 768), (768, 1))
    assert_size_stride(arg196_1, (384, ), (1, ))
    assert_size_stride(arg197_1, (384, 768), (768, 1))
    assert_size_stride(arg198_1, (384, ), (1, ))
    assert_size_stride(arg199_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg200_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg201_1, (54, 384), (384, 1))
    assert_size_stride(arg202_1, (54, ), (1, ))
    assert_size_stride(arg203_1, (384, 768), (768, 1))
    assert_size_stride(arg204_1, (384, ), (1, ))
    assert_size_stride(arg205_1, (768, 768), (768, 1))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (768, ), (1, ))
    assert_size_stride(arg209_1, (3072, 768), (768, 1))
    assert_size_stride(arg210_1, (3072, ), (1, ))
    assert_size_stride(arg211_1, (768, 3072), (3072, 1))
    assert_size_stride(arg212_1, (768, ), (1, ))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (768, ), (1, ))
    assert_size_stride(arg215_1, (384, 768), (768, 1))
    assert_size_stride(arg216_1, (384, ), (1, ))
    assert_size_stride(arg217_1, (384, 768), (768, 1))
    assert_size_stride(arg218_1, (384, ), (1, ))
    assert_size_stride(arg219_1, (384, 768), (768, 1))
    assert_size_stride(arg220_1, (384, ), (1, ))
    assert_size_stride(arg221_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg222_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg223_1, (54, 384), (384, 1))
    assert_size_stride(arg224_1, (54, ), (1, ))
    assert_size_stride(arg225_1, (384, 768), (768, 1))
    assert_size_stride(arg226_1, (384, ), (1, ))
    assert_size_stride(arg227_1, (768, 768), (768, 1))
    assert_size_stride(arg228_1, (768, ), (1, ))
    assert_size_stride(arg229_1, (768, ), (1, ))
    assert_size_stride(arg230_1, (768, ), (1, ))
    assert_size_stride(arg231_1, (3072, 768), (768, 1))
    assert_size_stride(arg232_1, (3072, ), (1, ))
    assert_size_stride(arg233_1, (768, 3072), (3072, 1))
    assert_size_stride(arg234_1, (768, ), (1, ))
    assert_size_stride(arg235_1, (768, ), (1, ))
    assert_size_stride(arg236_1, (768, ), (1, ))
    assert_size_stride(arg237_1, (384, 768), (768, 1))
    assert_size_stride(arg238_1, (384, ), (1, ))
    assert_size_stride(arg239_1, (384, 768), (768, 1))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (384, 768), (768, 1))
    assert_size_stride(arg242_1, (384, ), (1, ))
    assert_size_stride(arg243_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg244_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg245_1, (54, 384), (384, 1))
    assert_size_stride(arg246_1, (54, ), (1, ))
    assert_size_stride(arg247_1, (384, 768), (768, 1))
    assert_size_stride(arg248_1, (384, ), (1, ))
    assert_size_stride(arg249_1, (768, 768), (768, 1))
    assert_size_stride(arg250_1, (768, ), (1, ))
    assert_size_stride(arg251_1, (768, ), (1, ))
    assert_size_stride(arg252_1, (768, ), (1, ))
    assert_size_stride(arg253_1, (3072, 768), (768, 1))
    assert_size_stride(arg254_1, (3072, ), (1, ))
    assert_size_stride(arg255_1, (768, 3072), (3072, 1))
    assert_size_stride(arg256_1, (768, ), (1, ))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (768, ), (1, ))
    assert_size_stride(arg259_1, (384, 768), (768, 1))
    assert_size_stride(arg260_1, (384, ), (1, ))
    assert_size_stride(arg261_1, (384, 768), (768, 1))
    assert_size_stride(arg262_1, (384, ), (1, ))
    assert_size_stride(arg263_1, (384, 768), (768, 1))
    assert_size_stride(arg264_1, (384, ), (1, ))
    assert_size_stride(arg265_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg266_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg267_1, (54, 384), (384, 1))
    assert_size_stride(arg268_1, (54, ), (1, ))
    assert_size_stride(arg269_1, (384, 768), (768, 1))
    assert_size_stride(arg270_1, (384, ), (1, ))
    assert_size_stride(arg271_1, (768, 768), (768, 1))
    assert_size_stride(arg272_1, (768, ), (1, ))
    assert_size_stride(arg273_1, (768, ), (1, ))
    assert_size_stride(arg274_1, (768, ), (1, ))
    assert_size_stride(arg275_1, (3072, 768), (768, 1))
    assert_size_stride(arg276_1, (3072, ), (1, ))
    assert_size_stride(arg277_1, (768, 3072), (3072, 1))
    assert_size_stride(arg278_1, (768, ), (1, ))
    assert_size_stride(arg279_1, (768, ), (1, ))
    assert_size_stride(arg280_1, (768, ), (1, ))
    assert_size_stride(arg281_1, (768, 768), (768, 1))
    assert_size_stride(arg282_1, (768, ), (1, ))
    assert_size_stride(arg283_1, (768, ), (1, ))
    assert_size_stride(arg284_1, (768, ), (1, ))
    assert_size_stride(arg285_1, (30522, 768), (768, 1))
    assert_size_stride(arg286_1, (30522, ), (1, ))
    assert_size_stride(arg287_1, (1, 512), (512, 1))
    assert_size_stride(arg288_1, (1, 512), (512, 1))
    assert_size_stride(arg289_1, (1, 512), (512, 1))
    assert_size_stride(arg290_1, (1, 512), (512, 1))
    buf0 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf4 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_0(c_void_p(arg289_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg12_1
    del arg13_1
    del arg14_1
    del arg15_1
    del arg16_1
    del arg287_1
    del arg288_1
    del arg289_1
    buf5 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf5)
    del arg17_1
    del arg18_1
    buf6 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_key_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg19_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf6)
    del arg19_1
    del arg20_1
    buf7 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_value_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg22_1, reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf7)
    del arg21_1
    del arg22_1
    buf8 = empty_strided((1, 6, 512, 64), (196608, 64, 384, 1), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf6, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf6  # reuse
    buf10 = reinterpret_tensor(buf7, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf7  # reuse
    cpp_fused_1(c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf8.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf11 = aten._scaled_dot_product_flash_attention(buf8, buf9, buf10, scale=0.125)
    del buf10
    del buf8
    buf12 = buf11[0]
    del buf11
    buf19 = reinterpret_tensor(buf9, (512, 384), (384, 1), 0); del buf9  # reuse
    # Source Nodes: [conv_out_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg28_1, reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf19)
    del arg27_1
    del arg28_1
    buf20 = reinterpret_tensor(buf0, (1, 768, 512), (393216, 512, 1), 0); del buf0  # reuse
    cpp_fused_convolution_2(c_void_p(buf4.data_ptr()), c_void_p(buf20.data_ptr()))
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(buf20, arg23_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf21, (1, 768, 512), (393216, 512, 1))
    del arg23_1
    # Source Nodes: [x_1], Original ATen: [aten.convolution]
    buf22 = extern_kernels.convolution(buf21, arg24_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf22, (1, 384, 512), (196608, 512, 1))
    del arg24_1
    buf23 = reinterpret_tensor(buf5, (1, 512, 384), (196608, 384, 1), 0); del buf5  # reuse
    cpp_fused_mul_3(c_void_p(buf23.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(arg0_1.data_ptr()))
    del arg0_1
    buf24 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf23, (512, 384), (384, 1), 0), reinterpret_tensor(arg25_1, (384, 54), (1, 384), 0), out=buf24)
    del arg25_1
    buf25 = empty_strided((3072, 1, 1), (1, 3072, 3072), device='cpu', dtype=torch.float32)
    buf26 = reinterpret_tensor(buf24, (3072, 9, 1), (9, 1, 27648), 0); del buf24  # reuse
    buf27 = empty_strided((3072, 1, 1), (1, 3072, 3072), device='cpu', dtype=torch.float32)
    buf28 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    buf29 = buf26; del buf26  # reuse
    cpp_fused__softmax_clone_4(c_void_p(buf29.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    del arg26_1
    buf30 = reinterpret_tensor(buf19, (3072, 64, 1), (64, 1, 1), 0); del buf19  # reuse
    # Source Nodes: [conv_kernel_layer_2, conv_out_layer_6], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf28, (3072, 64, 9), (576, 9, 1), 0), buf29, out=buf30)
    buf31 = reinterpret_tensor(buf21, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf21  # reuse
    cpp_fused_cat_5(c_void_p(buf12.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    buf32 = reinterpret_tensor(buf20, (512, 768), (768, 1), 0); del buf20  # reuse
    # Source Nodes: [hidden_states_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg30_1, reinterpret_tensor(buf31, (512, 768), (768, 1), 0), reinterpret_tensor(arg29_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf32)
    del arg29_1
    del arg30_1
    buf33 = buf2; del buf2  # reuse
    buf34 = buf1; del buf1  # reuse
    buf36 = reinterpret_tensor(buf31, (1, 512, 768), (393216, 768, 1), 0); del buf31  # reuse
    cpp_fused_add_native_layer_norm_6(c_void_p(buf32.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()))
    del arg31_1
    del arg32_1
    buf37 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg34_1, reinterpret_tensor(buf36, (512, 768), (768, 1), 0), reinterpret_tensor(arg33_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf37)
    del arg33_1
    del arg34_1
    buf38 = reinterpret_tensor(buf37, (1, 512, 3072), (1572864, 3072, 1), 0); del buf37  # reuse
    cpp_fused_gelu_7(c_void_p(buf38.data_ptr()))
    buf39 = reinterpret_tensor(buf4, (512, 768), (768, 1), 0); del buf4  # reuse
    # Source Nodes: [hidden_states_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg36_1, reinterpret_tensor(buf38, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg35_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf39)
    del arg35_1
    del arg36_1
    buf40 = buf34; del buf34  # reuse
    buf41 = buf33; del buf33  # reuse
    buf43 = reinterpret_tensor(buf32, (1, 512, 768), (393216, 768, 1), 0); del buf32  # reuse
    cpp_fused_add_native_layer_norm_8(c_void_p(buf39.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()))
    del arg37_1
    del arg38_1
    del buf36
    buf44 = reinterpret_tensor(buf30, (512, 384), (384, 1), 0); del buf30  # reuse
    # Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg40_1, reinterpret_tensor(buf43, (512, 768), (768, 1), 0), reinterpret_tensor(arg39_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf44)
    del arg39_1
    del arg40_1
    buf45 = reinterpret_tensor(buf12, (512, 384), (384, 1), 0); del buf12  # reuse
    # Source Nodes: [mixed_key_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg42_1, reinterpret_tensor(buf43, (512, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf45)
    del arg41_1
    del arg42_1
    buf46 = reinterpret_tensor(buf23, (512, 384), (384, 1), 0); del buf23  # reuse
    # Source Nodes: [mixed_value_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg44_1, reinterpret_tensor(buf43, (512, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf46)
    del arg43_1
    del arg44_1
    buf47 = reinterpret_tensor(buf22, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf22  # reuse
    buf48 = reinterpret_tensor(buf45, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf45  # reuse
    buf49 = reinterpret_tensor(buf46, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf46  # reuse
    cpp_fused_9(c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf47.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf50 = aten._scaled_dot_product_flash_attention(buf47, buf48, buf49, scale=0.125)
    del buf47
    del buf48
    buf51 = buf50[0]
    del buf50
    buf58 = reinterpret_tensor(buf49, (512, 384), (384, 1), 0); del buf49  # reuse
    # Source Nodes: [conv_out_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg50_1, reinterpret_tensor(buf43, (512, 768), (768, 1), 0), reinterpret_tensor(arg49_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf58)
    del arg49_1
    del arg50_1
    buf59 = reinterpret_tensor(buf39, (1, 768, 512), (393216, 512, 1), 0); del buf39  # reuse
    cpp_fused_convolution_10(c_void_p(buf43.data_ptr()), c_void_p(buf59.data_ptr()))
    # Source Nodes: [x_6], Original ATen: [aten.convolution]
    buf60 = extern_kernels.convolution(buf59, arg45_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf60, (1, 768, 512), (393216, 512, 1))
    del arg45_1
    # Source Nodes: [x_7], Original ATen: [aten.convolution]
    buf61 = extern_kernels.convolution(buf60, arg46_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf61, (1, 384, 512), (196608, 512, 1))
    del arg46_1
    buf62 = reinterpret_tensor(buf44, (1, 512, 384), (196608, 384, 1), 0); del buf44  # reuse
    cpp_fused_mul_11(c_void_p(buf62.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg1_1
    buf63 = reinterpret_tensor(buf29, (512, 54), (54, 1), 0); del buf29  # reuse
    # Source Nodes: [conv_kernel_layer_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (512, 384), (384, 1), 0), reinterpret_tensor(arg47_1, (384, 54), (1, 384), 0), out=buf63)
    del arg47_1
    buf64 = buf27; del buf27  # reuse
    buf65 = reinterpret_tensor(buf63, (3072, 9, 1), (9, 1, 27648), 0); del buf63  # reuse
    buf66 = buf25; del buf25  # reuse
    buf67 = buf28; del buf28  # reuse
    buf68 = buf65; del buf65  # reuse
    cpp_fused__softmax_clone_12(c_void_p(buf68.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    del arg48_1
    buf69 = reinterpret_tensor(buf58, (3072, 64, 1), (64, 1, 1), 0); del buf58  # reuse
    # Source Nodes: [conv_kernel_layer_5, conv_out_layer_14], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf67, (3072, 64, 9), (576, 9, 1), 0), buf68, out=buf69)
    buf70 = reinterpret_tensor(buf60, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf60  # reuse
    cpp_fused_cat_13(c_void_p(buf51.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    buf71 = reinterpret_tensor(buf59, (512, 768), (768, 1), 0); del buf59  # reuse
    # Source Nodes: [hidden_states_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg52_1, reinterpret_tensor(buf70, (512, 768), (768, 1), 0), reinterpret_tensor(arg51_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf71)
    del arg51_1
    del arg52_1
    buf72 = buf41; del buf41  # reuse
    buf73 = buf40; del buf40  # reuse
    buf75 = reinterpret_tensor(buf70, (1, 512, 768), (393216, 768, 1), 0); del buf70  # reuse
    cpp_fused_add_native_layer_norm_14(c_void_p(buf71.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()))
    del arg53_1
    del arg54_1
    buf76 = reinterpret_tensor(buf38, (512, 3072), (3072, 1), 0); del buf38  # reuse
    # Source Nodes: [hidden_states_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg56_1, reinterpret_tensor(buf75, (512, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf76)
    del arg55_1
    del arg56_1
    buf77 = reinterpret_tensor(buf76, (1, 512, 3072), (1572864, 3072, 1), 0); del buf76  # reuse
    cpp_fused_gelu_15(c_void_p(buf77.data_ptr()))
    buf78 = buf71; del buf71  # reuse
    # Source Nodes: [hidden_states_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg58_1, reinterpret_tensor(buf77, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg57_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf78)
    del arg57_1
    del arg58_1
    buf79 = buf73; del buf73  # reuse
    buf80 = buf72; del buf72  # reuse
    buf82 = buf43; del buf43  # reuse
    cpp_fused_add_native_layer_norm_16(c_void_p(buf78.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()))
    del arg59_1
    del arg60_1
    del buf75
    buf83 = reinterpret_tensor(buf69, (512, 384), (384, 1), 0); del buf69  # reuse
    # Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg62_1, reinterpret_tensor(buf82, (512, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf83)
    del arg61_1
    del arg62_1
    buf84 = reinterpret_tensor(buf51, (512, 384), (384, 1), 0); del buf51  # reuse
    # Source Nodes: [mixed_key_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg64_1, reinterpret_tensor(buf82, (512, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf84)
    del arg63_1
    del arg64_1
    buf85 = reinterpret_tensor(buf62, (512, 384), (384, 1), 0); del buf62  # reuse
    # Source Nodes: [mixed_value_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg66_1, reinterpret_tensor(buf82, (512, 768), (768, 1), 0), reinterpret_tensor(arg65_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf85)
    del arg65_1
    del arg66_1
    buf86 = reinterpret_tensor(buf61, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf61  # reuse
    buf87 = reinterpret_tensor(buf84, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf84  # reuse
    buf88 = reinterpret_tensor(buf85, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf85  # reuse
    cpp_fused_17(c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf86.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf89 = aten._scaled_dot_product_flash_attention(buf86, buf87, buf88, scale=0.125)
    del buf86
    del buf87
    buf90 = buf89[0]
    del buf89
    buf97 = reinterpret_tensor(buf88, (512, 384), (384, 1), 0); del buf88  # reuse
    # Source Nodes: [conv_out_layer_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg72_1, reinterpret_tensor(buf82, (512, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf97)
    del arg71_1
    del arg72_1
    buf98 = reinterpret_tensor(buf78, (1, 768, 512), (393216, 512, 1), 0); del buf78  # reuse
    cpp_fused_convolution_18(c_void_p(buf82.data_ptr()), c_void_p(buf98.data_ptr()))
    # Source Nodes: [x_12], Original ATen: [aten.convolution]
    buf99 = extern_kernels.convolution(buf98, arg67_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf99, (1, 768, 512), (393216, 512, 1))
    del arg67_1
    # Source Nodes: [x_13], Original ATen: [aten.convolution]
    buf100 = extern_kernels.convolution(buf99, arg68_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf100, (1, 384, 512), (196608, 512, 1))
    del arg68_1
    buf101 = reinterpret_tensor(buf83, (1, 512, 384), (196608, 384, 1), 0); del buf83  # reuse
    cpp_fused_mul_19(c_void_p(buf101.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(arg2_1.data_ptr()))
    del arg2_1
    buf102 = reinterpret_tensor(buf68, (512, 54), (54, 1), 0); del buf68  # reuse
    # Source Nodes: [conv_kernel_layer_6], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf101, (512, 384), (384, 1), 0), reinterpret_tensor(arg69_1, (384, 54), (1, 384), 0), out=buf102)
    del arg69_1
    buf103 = buf66; del buf66  # reuse
    buf104 = reinterpret_tensor(buf102, (3072, 9, 1), (9, 1, 27648), 0); del buf102  # reuse
    buf105 = buf64; del buf64  # reuse
    buf106 = buf67; del buf67  # reuse
    buf107 = buf104; del buf104  # reuse
    cpp_fused__softmax_clone_20(c_void_p(buf107.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del arg70_1
    buf108 = reinterpret_tensor(buf97, (3072, 64, 1), (64, 1, 1), 0); del buf97  # reuse
    # Source Nodes: [conv_kernel_layer_8, conv_out_layer_22], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf106, (3072, 64, 9), (576, 9, 1), 0), buf107, out=buf108)
    buf109 = reinterpret_tensor(buf99, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf99  # reuse
    cpp_fused_cat_21(c_void_p(buf90.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    buf110 = reinterpret_tensor(buf98, (512, 768), (768, 1), 0); del buf98  # reuse
    # Source Nodes: [hidden_states_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg74_1, reinterpret_tensor(buf109, (512, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf110)
    del arg73_1
    del arg74_1
    buf111 = buf80; del buf80  # reuse
    buf112 = buf79; del buf79  # reuse
    buf114 = reinterpret_tensor(buf109, (1, 512, 768), (393216, 768, 1), 0); del buf109  # reuse
    cpp_fused_add_native_layer_norm_22(c_void_p(buf110.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()))
    del arg75_1
    del arg76_1
    buf115 = reinterpret_tensor(buf77, (512, 3072), (3072, 1), 0); del buf77  # reuse
    # Source Nodes: [hidden_states_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg78_1, reinterpret_tensor(buf114, (512, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf115)
    del arg77_1
    del arg78_1
    buf116 = reinterpret_tensor(buf115, (1, 512, 3072), (1572864, 3072, 1), 0); del buf115  # reuse
    cpp_fused_gelu_23(c_void_p(buf116.data_ptr()))
    buf117 = reinterpret_tensor(buf82, (512, 768), (768, 1), 0); del buf82  # reuse
    # Source Nodes: [hidden_states_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg80_1, reinterpret_tensor(buf116, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg79_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf117)
    del arg79_1
    del arg80_1
    buf118 = buf112; del buf112  # reuse
    buf119 = buf111; del buf111  # reuse
    buf121 = reinterpret_tensor(buf110, (1, 512, 768), (393216, 768, 1), 0); del buf110  # reuse
    cpp_fused_add_native_layer_norm_24(c_void_p(buf117.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()))
    del arg81_1
    del arg82_1
    del buf114
    buf122 = reinterpret_tensor(buf90, (512, 384), (384, 1), 0); del buf90  # reuse
    # Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg84_1, reinterpret_tensor(buf121, (512, 768), (768, 1), 0), reinterpret_tensor(arg83_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf122)
    del arg83_1
    del arg84_1
    buf123 = reinterpret_tensor(buf108, (512, 384), (384, 1), 0); del buf108  # reuse
    # Source Nodes: [mixed_key_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg86_1, reinterpret_tensor(buf121, (512, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf123)
    del arg85_1
    del arg86_1
    buf124 = reinterpret_tensor(buf101, (512, 384), (384, 1), 0); del buf101  # reuse
    # Source Nodes: [mixed_value_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg88_1, reinterpret_tensor(buf121, (512, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf124)
    del arg87_1
    del arg88_1
    buf125 = reinterpret_tensor(buf100, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf100  # reuse
    buf126 = reinterpret_tensor(buf123, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf123  # reuse
    buf127 = reinterpret_tensor(buf124, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf124  # reuse
    cpp_fused_25(c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf125.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf128 = aten._scaled_dot_product_flash_attention(buf125, buf126, buf127, scale=0.125)
    del buf125
    del buf126
    buf129 = buf128[0]
    del buf128
    buf136 = reinterpret_tensor(buf127, (512, 384), (384, 1), 0); del buf127  # reuse
    # Source Nodes: [conv_out_layer_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg94_1, reinterpret_tensor(buf121, (512, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf136)
    del arg93_1
    del arg94_1
    buf137 = reinterpret_tensor(buf117, (1, 768, 512), (393216, 512, 1), 0); del buf117  # reuse
    cpp_fused_convolution_26(c_void_p(buf121.data_ptr()), c_void_p(buf137.data_ptr()))
    # Source Nodes: [x_18], Original ATen: [aten.convolution]
    buf138 = extern_kernels.convolution(buf137, arg89_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf138, (1, 768, 512), (393216, 512, 1))
    del arg89_1
    # Source Nodes: [x_19], Original ATen: [aten.convolution]
    buf139 = extern_kernels.convolution(buf138, arg90_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf139, (1, 384, 512), (196608, 512, 1))
    del arg90_1
    buf140 = reinterpret_tensor(buf122, (1, 512, 384), (196608, 384, 1), 0); del buf122  # reuse
    cpp_fused_mul_27(c_void_p(buf140.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg3_1.data_ptr()))
    del arg3_1
    buf141 = reinterpret_tensor(buf107, (512, 54), (54, 1), 0); del buf107  # reuse
    # Source Nodes: [conv_kernel_layer_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (512, 384), (384, 1), 0), reinterpret_tensor(arg91_1, (384, 54), (1, 384), 0), out=buf141)
    del arg91_1
    buf142 = buf105; del buf105  # reuse
    buf143 = reinterpret_tensor(buf141, (3072, 9, 1), (9, 1, 27648), 0); del buf141  # reuse
    buf144 = buf103; del buf103  # reuse
    buf145 = buf106; del buf106  # reuse
    buf146 = buf143; del buf143  # reuse
    cpp_fused__softmax_clone_28(c_void_p(buf146.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del arg92_1
    buf147 = reinterpret_tensor(buf136, (3072, 64, 1), (64, 1, 1), 0); del buf136  # reuse
    # Source Nodes: [conv_kernel_layer_11, conv_out_layer_30], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf145, (3072, 64, 9), (576, 9, 1), 0), buf146, out=buf147)
    buf148 = reinterpret_tensor(buf138, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf138  # reuse
    cpp_fused_cat_29(c_void_p(buf129.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    buf149 = reinterpret_tensor(buf137, (512, 768), (768, 1), 0); del buf137  # reuse
    # Source Nodes: [hidden_states_28], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg96_1, reinterpret_tensor(buf148, (512, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf149)
    del arg95_1
    del arg96_1
    buf150 = buf119; del buf119  # reuse
    buf151 = buf118; del buf118  # reuse
    buf153 = reinterpret_tensor(buf148, (1, 512, 768), (393216, 768, 1), 0); del buf148  # reuse
    cpp_fused_add_native_layer_norm_30(c_void_p(buf149.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()))
    del arg97_1
    del arg98_1
    buf154 = reinterpret_tensor(buf116, (512, 3072), (3072, 1), 0); del buf116  # reuse
    # Source Nodes: [hidden_states_31], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg100_1, reinterpret_tensor(buf153, (512, 768), (768, 1), 0), reinterpret_tensor(arg99_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf154)
    del arg100_1
    del arg99_1
    buf155 = reinterpret_tensor(buf154, (1, 512, 3072), (1572864, 3072, 1), 0); del buf154  # reuse
    cpp_fused_gelu_31(c_void_p(buf155.data_ptr()))
    buf156 = buf149; del buf149  # reuse
    # Source Nodes: [hidden_states_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg102_1, reinterpret_tensor(buf155, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg101_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf156)
    del arg101_1
    del arg102_1
    buf157 = buf151; del buf151  # reuse
    buf158 = buf150; del buf150  # reuse
    buf160 = buf121; del buf121  # reuse
    cpp_fused_add_native_layer_norm_32(c_void_p(buf156.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()))
    del arg103_1
    del arg104_1
    del buf153
    buf161 = reinterpret_tensor(buf147, (512, 384), (384, 1), 0); del buf147  # reuse
    # Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg106_1, reinterpret_tensor(buf160, (512, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf161)
    del arg105_1
    del arg106_1
    buf162 = reinterpret_tensor(buf129, (512, 384), (384, 1), 0); del buf129  # reuse
    # Source Nodes: [mixed_key_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg108_1, reinterpret_tensor(buf160, (512, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf162)
    del arg107_1
    del arg108_1
    buf163 = reinterpret_tensor(buf140, (512, 384), (384, 1), 0); del buf140  # reuse
    # Source Nodes: [mixed_value_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg110_1, reinterpret_tensor(buf160, (512, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf163)
    del arg109_1
    del arg110_1
    buf164 = reinterpret_tensor(buf139, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf139  # reuse
    buf165 = reinterpret_tensor(buf162, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf162  # reuse
    buf166 = reinterpret_tensor(buf163, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf163  # reuse
    cpp_fused_33(c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf164.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf167 = aten._scaled_dot_product_flash_attention(buf164, buf165, buf166, scale=0.125)
    del buf164
    del buf165
    buf168 = buf167[0]
    del buf167
    buf175 = reinterpret_tensor(buf166, (512, 384), (384, 1), 0); del buf166  # reuse
    # Source Nodes: [conv_out_layer_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg116_1, reinterpret_tensor(buf160, (512, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf175)
    del arg115_1
    del arg116_1
    buf176 = reinterpret_tensor(buf156, (1, 768, 512), (393216, 512, 1), 0); del buf156  # reuse
    cpp_fused_convolution_34(c_void_p(buf160.data_ptr()), c_void_p(buf176.data_ptr()))
    # Source Nodes: [x_24], Original ATen: [aten.convolution]
    buf177 = extern_kernels.convolution(buf176, arg111_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf177, (1, 768, 512), (393216, 512, 1))
    del arg111_1
    # Source Nodes: [x_25], Original ATen: [aten.convolution]
    buf178 = extern_kernels.convolution(buf177, arg112_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf178, (1, 384, 512), (196608, 512, 1))
    del arg112_1
    buf179 = reinterpret_tensor(buf161, (1, 512, 384), (196608, 384, 1), 0); del buf161  # reuse
    cpp_fused_mul_35(c_void_p(buf179.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(arg4_1.data_ptr()))
    del arg4_1
    buf180 = reinterpret_tensor(buf146, (512, 54), (54, 1), 0); del buf146  # reuse
    # Source Nodes: [conv_kernel_layer_12], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf179, (512, 384), (384, 1), 0), reinterpret_tensor(arg113_1, (384, 54), (1, 384), 0), out=buf180)
    del arg113_1
    buf181 = buf144; del buf144  # reuse
    buf182 = reinterpret_tensor(buf180, (3072, 9, 1), (9, 1, 27648), 0); del buf180  # reuse
    buf183 = buf142; del buf142  # reuse
    buf184 = buf145; del buf145  # reuse
    buf185 = buf182; del buf182  # reuse
    cpp_fused__softmax_clone_36(c_void_p(buf185.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()))
    del arg114_1
    buf186 = reinterpret_tensor(buf175, (3072, 64, 1), (64, 1, 1), 0); del buf175  # reuse
    # Source Nodes: [conv_kernel_layer_14, conv_out_layer_38], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf184, (3072, 64, 9), (576, 9, 1), 0), buf185, out=buf186)
    buf187 = reinterpret_tensor(buf177, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf177  # reuse
    cpp_fused_cat_37(c_void_p(buf168.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()))
    buf188 = reinterpret_tensor(buf176, (512, 768), (768, 1), 0); del buf176  # reuse
    # Source Nodes: [hidden_states_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg118_1, reinterpret_tensor(buf187, (512, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf188)
    del arg117_1
    del arg118_1
    buf189 = buf158; del buf158  # reuse
    buf190 = buf157; del buf157  # reuse
    buf192 = reinterpret_tensor(buf187, (1, 512, 768), (393216, 768, 1), 0); del buf187  # reuse
    cpp_fused_add_native_layer_norm_38(c_void_p(buf188.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf192.data_ptr()))
    del arg119_1
    del arg120_1
    buf193 = reinterpret_tensor(buf155, (512, 3072), (3072, 1), 0); del buf155  # reuse
    # Source Nodes: [hidden_states_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg122_1, reinterpret_tensor(buf192, (512, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf193)
    del arg121_1
    del arg122_1
    buf194 = reinterpret_tensor(buf193, (1, 512, 3072), (1572864, 3072, 1), 0); del buf193  # reuse
    cpp_fused_gelu_39(c_void_p(buf194.data_ptr()))
    buf195 = buf188; del buf188  # reuse
    # Source Nodes: [hidden_states_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg124_1, reinterpret_tensor(buf194, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg123_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf195)
    del arg123_1
    del arg124_1
    buf196 = buf190; del buf190  # reuse
    buf197 = buf189; del buf189  # reuse
    buf199 = buf160; del buf160  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf195.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()))
    del arg125_1
    del arg126_1
    del buf192
    buf200 = reinterpret_tensor(buf186, (512, 384), (384, 1), 0); del buf186  # reuse
    # Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg128_1, reinterpret_tensor(buf199, (512, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf200)
    del arg127_1
    del arg128_1
    buf201 = reinterpret_tensor(buf168, (512, 384), (384, 1), 0); del buf168  # reuse
    # Source Nodes: [mixed_key_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg130_1, reinterpret_tensor(buf199, (512, 768), (768, 1), 0), reinterpret_tensor(arg129_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf201)
    del arg129_1
    del arg130_1
    buf202 = reinterpret_tensor(buf179, (512, 384), (384, 1), 0); del buf179  # reuse
    # Source Nodes: [mixed_value_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg132_1, reinterpret_tensor(buf199, (512, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf202)
    del arg131_1
    del arg132_1
    buf203 = reinterpret_tensor(buf178, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf178  # reuse
    buf204 = reinterpret_tensor(buf201, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf201  # reuse
    buf205 = reinterpret_tensor(buf202, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf202  # reuse
    cpp_fused_41(c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf203.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf206 = aten._scaled_dot_product_flash_attention(buf203, buf204, buf205, scale=0.125)
    del buf203
    del buf204
    buf207 = buf206[0]
    del buf206
    buf214 = reinterpret_tensor(buf205, (512, 384), (384, 1), 0); del buf205  # reuse
    # Source Nodes: [conv_out_layer_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf199, (512, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf214)
    del arg137_1
    del arg138_1
    buf215 = reinterpret_tensor(buf195, (1, 768, 512), (393216, 512, 1), 0); del buf195  # reuse
    cpp_fused_convolution_42(c_void_p(buf199.data_ptr()), c_void_p(buf215.data_ptr()))
    # Source Nodes: [x_30], Original ATen: [aten.convolution]
    buf216 = extern_kernels.convolution(buf215, arg133_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf216, (1, 768, 512), (393216, 512, 1))
    del arg133_1
    # Source Nodes: [x_31], Original ATen: [aten.convolution]
    buf217 = extern_kernels.convolution(buf216, arg134_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf217, (1, 384, 512), (196608, 512, 1))
    del arg134_1
    buf218 = reinterpret_tensor(buf200, (1, 512, 384), (196608, 384, 1), 0); del buf200  # reuse
    cpp_fused_mul_43(c_void_p(buf218.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg5_1
    buf219 = reinterpret_tensor(buf185, (512, 54), (54, 1), 0); del buf185  # reuse
    # Source Nodes: [conv_kernel_layer_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (512, 384), (384, 1), 0), reinterpret_tensor(arg135_1, (384, 54), (1, 384), 0), out=buf219)
    del arg135_1
    buf220 = buf183; del buf183  # reuse
    buf221 = reinterpret_tensor(buf219, (3072, 9, 1), (9, 1, 27648), 0); del buf219  # reuse
    buf222 = buf181; del buf181  # reuse
    buf223 = buf184; del buf184  # reuse
    buf224 = buf221; del buf221  # reuse
    cpp_fused__softmax_clone_44(c_void_p(buf224.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    del arg136_1
    buf225 = reinterpret_tensor(buf214, (3072, 64, 1), (64, 1, 1), 0); del buf214  # reuse
    # Source Nodes: [conv_kernel_layer_17, conv_out_layer_46], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf223, (3072, 64, 9), (576, 9, 1), 0), buf224, out=buf225)
    buf226 = reinterpret_tensor(buf216, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf216  # reuse
    cpp_fused_cat_45(c_void_p(buf207.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    buf227 = reinterpret_tensor(buf215, (512, 768), (768, 1), 0); del buf215  # reuse
    # Source Nodes: [hidden_states_46], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg140_1, reinterpret_tensor(buf226, (512, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf227)
    del arg139_1
    del arg140_1
    buf228 = buf197; del buf197  # reuse
    buf229 = buf196; del buf196  # reuse
    buf231 = reinterpret_tensor(buf226, (1, 512, 768), (393216, 768, 1), 0); del buf226  # reuse
    cpp_fused_add_native_layer_norm_46(c_void_p(buf227.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()))
    del arg141_1
    del arg142_1
    buf232 = reinterpret_tensor(buf194, (512, 3072), (3072, 1), 0); del buf194  # reuse
    # Source Nodes: [hidden_states_49], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg144_1, reinterpret_tensor(buf231, (512, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf232)
    del arg143_1
    del arg144_1
    buf233 = reinterpret_tensor(buf232, (1, 512, 3072), (1572864, 3072, 1), 0); del buf232  # reuse
    cpp_fused_gelu_47(c_void_p(buf233.data_ptr()))
    buf234 = buf227; del buf227  # reuse
    # Source Nodes: [hidden_states_51], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg146_1, reinterpret_tensor(buf233, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg145_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf234)
    del arg145_1
    del arg146_1
    buf235 = buf229; del buf229  # reuse
    buf236 = buf228; del buf228  # reuse
    buf238 = buf199; del buf199  # reuse
    cpp_fused_add_native_layer_norm_48(c_void_p(buf234.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()))
    del arg147_1
    del arg148_1
    del buf231
    buf239 = reinterpret_tensor(buf225, (512, 384), (384, 1), 0); del buf225  # reuse
    # Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg150_1, reinterpret_tensor(buf238, (512, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf239)
    del arg149_1
    del arg150_1
    buf240 = reinterpret_tensor(buf207, (512, 384), (384, 1), 0); del buf207  # reuse
    # Source Nodes: [mixed_key_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg152_1, reinterpret_tensor(buf238, (512, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf240)
    del arg151_1
    del arg152_1
    buf241 = reinterpret_tensor(buf218, (512, 384), (384, 1), 0); del buf218  # reuse
    # Source Nodes: [mixed_value_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg154_1, reinterpret_tensor(buf238, (512, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf241)
    del arg153_1
    del arg154_1
    buf242 = reinterpret_tensor(buf217, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf217  # reuse
    buf243 = reinterpret_tensor(buf240, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf240  # reuse
    buf244 = reinterpret_tensor(buf241, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf241  # reuse
    cpp_fused_49(c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf242.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf245 = aten._scaled_dot_product_flash_attention(buf242, buf243, buf244, scale=0.125)
    del buf242
    del buf243
    buf246 = buf245[0]
    del buf245
    buf253 = reinterpret_tensor(buf244, (512, 384), (384, 1), 0); del buf244  # reuse
    # Source Nodes: [conv_out_layer_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg160_1, reinterpret_tensor(buf238, (512, 768), (768, 1), 0), reinterpret_tensor(arg159_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf253)
    del arg159_1
    del arg160_1
    buf254 = reinterpret_tensor(buf234, (1, 768, 512), (393216, 512, 1), 0); del buf234  # reuse
    cpp_fused_convolution_50(c_void_p(buf238.data_ptr()), c_void_p(buf254.data_ptr()))
    # Source Nodes: [x_36], Original ATen: [aten.convolution]
    buf255 = extern_kernels.convolution(buf254, arg155_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf255, (1, 768, 512), (393216, 512, 1))
    del arg155_1
    # Source Nodes: [x_37], Original ATen: [aten.convolution]
    buf256 = extern_kernels.convolution(buf255, arg156_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf256, (1, 384, 512), (196608, 512, 1))
    del arg156_1
    buf257 = reinterpret_tensor(buf239, (1, 512, 384), (196608, 384, 1), 0); del buf239  # reuse
    cpp_fused_mul_51(c_void_p(buf257.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(arg6_1.data_ptr()))
    del arg6_1
    buf258 = reinterpret_tensor(buf224, (512, 54), (54, 1), 0); del buf224  # reuse
    # Source Nodes: [conv_kernel_layer_18], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (512, 384), (384, 1), 0), reinterpret_tensor(arg157_1, (384, 54), (1, 384), 0), out=buf258)
    del arg157_1
    buf259 = buf222; del buf222  # reuse
    buf260 = reinterpret_tensor(buf258, (3072, 9, 1), (9, 1, 27648), 0); del buf258  # reuse
    buf261 = buf220; del buf220  # reuse
    buf262 = buf223; del buf223  # reuse
    buf263 = buf260; del buf260  # reuse
    cpp_fused__softmax_clone_52(c_void_p(buf263.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()))
    del arg158_1
    buf264 = reinterpret_tensor(buf253, (3072, 64, 1), (64, 1, 1), 0); del buf253  # reuse
    # Source Nodes: [conv_kernel_layer_20, conv_out_layer_54], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf262, (3072, 64, 9), (576, 9, 1), 0), buf263, out=buf264)
    buf265 = reinterpret_tensor(buf255, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf255  # reuse
    cpp_fused_cat_53(c_void_p(buf246.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    buf266 = reinterpret_tensor(buf254, (512, 768), (768, 1), 0); del buf254  # reuse
    # Source Nodes: [hidden_states_55], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg162_1, reinterpret_tensor(buf265, (512, 768), (768, 1), 0), reinterpret_tensor(arg161_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf266)
    del arg161_1
    del arg162_1
    buf267 = buf236; del buf236  # reuse
    buf268 = buf235; del buf235  # reuse
    buf270 = reinterpret_tensor(buf265, (1, 512, 768), (393216, 768, 1), 0); del buf265  # reuse
    cpp_fused_add_native_layer_norm_54(c_void_p(buf266.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()))
    del arg163_1
    del arg164_1
    buf271 = reinterpret_tensor(buf233, (512, 3072), (3072, 1), 0); del buf233  # reuse
    # Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg166_1, reinterpret_tensor(buf270, (512, 768), (768, 1), 0), reinterpret_tensor(arg165_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf271)
    del arg165_1
    del arg166_1
    buf272 = reinterpret_tensor(buf271, (1, 512, 3072), (1572864, 3072, 1), 0); del buf271  # reuse
    cpp_fused_gelu_55(c_void_p(buf272.data_ptr()))
    buf273 = buf266; del buf266  # reuse
    # Source Nodes: [hidden_states_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg168_1, reinterpret_tensor(buf272, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg167_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf273)
    del arg167_1
    del arg168_1
    buf274 = buf268; del buf268  # reuse
    buf275 = buf267; del buf267  # reuse
    buf277 = buf238; del buf238  # reuse
    cpp_fused_add_native_layer_norm_56(c_void_p(buf273.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf277.data_ptr()))
    del arg169_1
    del arg170_1
    del buf270
    buf278 = reinterpret_tensor(buf264, (512, 384), (384, 1), 0); del buf264  # reuse
    # Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg172_1, reinterpret_tensor(buf277, (512, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf278)
    del arg171_1
    del arg172_1
    buf279 = reinterpret_tensor(buf246, (512, 384), (384, 1), 0); del buf246  # reuse
    # Source Nodes: [mixed_key_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg174_1, reinterpret_tensor(buf277, (512, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf279)
    del arg173_1
    del arg174_1
    buf280 = reinterpret_tensor(buf257, (512, 384), (384, 1), 0); del buf257  # reuse
    # Source Nodes: [mixed_value_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg176_1, reinterpret_tensor(buf277, (512, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf280)
    del arg175_1
    del arg176_1
    buf281 = reinterpret_tensor(buf256, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf256  # reuse
    buf282 = reinterpret_tensor(buf279, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf279  # reuse
    buf283 = reinterpret_tensor(buf280, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf280  # reuse
    cpp_fused_57(c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf281.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf284 = aten._scaled_dot_product_flash_attention(buf281, buf282, buf283, scale=0.125)
    del buf281
    del buf282
    buf285 = buf284[0]
    del buf284
    buf292 = reinterpret_tensor(buf283, (512, 384), (384, 1), 0); del buf283  # reuse
    # Source Nodes: [conv_out_layer_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg182_1, reinterpret_tensor(buf277, (512, 768), (768, 1), 0), reinterpret_tensor(arg181_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf292)
    del arg181_1
    del arg182_1
    buf293 = reinterpret_tensor(buf273, (1, 768, 512), (393216, 512, 1), 0); del buf273  # reuse
    cpp_fused_convolution_58(c_void_p(buf277.data_ptr()), c_void_p(buf293.data_ptr()))
    # Source Nodes: [x_42], Original ATen: [aten.convolution]
    buf294 = extern_kernels.convolution(buf293, arg177_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf294, (1, 768, 512), (393216, 512, 1))
    del arg177_1
    # Source Nodes: [x_43], Original ATen: [aten.convolution]
    buf295 = extern_kernels.convolution(buf294, arg178_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf295, (1, 384, 512), (196608, 512, 1))
    del arg178_1
    buf296 = reinterpret_tensor(buf278, (1, 512, 384), (196608, 384, 1), 0); del buf278  # reuse
    cpp_fused_mul_59(c_void_p(buf296.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(arg7_1.data_ptr()))
    del arg7_1
    buf297 = reinterpret_tensor(buf263, (512, 54), (54, 1), 0); del buf263  # reuse
    # Source Nodes: [conv_kernel_layer_21], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf296, (512, 384), (384, 1), 0), reinterpret_tensor(arg179_1, (384, 54), (1, 384), 0), out=buf297)
    del arg179_1
    buf298 = buf261; del buf261  # reuse
    buf299 = reinterpret_tensor(buf297, (3072, 9, 1), (9, 1, 27648), 0); del buf297  # reuse
    buf300 = buf259; del buf259  # reuse
    buf301 = buf262; del buf262  # reuse
    buf302 = buf299; del buf299  # reuse
    cpp_fused__softmax_clone_60(c_void_p(buf302.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    del arg180_1
    buf303 = reinterpret_tensor(buf292, (3072, 64, 1), (64, 1, 1), 0); del buf292  # reuse
    # Source Nodes: [conv_kernel_layer_23, conv_out_layer_62], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf301, (3072, 64, 9), (576, 9, 1), 0), buf302, out=buf303)
    buf304 = reinterpret_tensor(buf294, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf294  # reuse
    cpp_fused_cat_61(c_void_p(buf285.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    buf305 = reinterpret_tensor(buf293, (512, 768), (768, 1), 0); del buf293  # reuse
    # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg184_1, reinterpret_tensor(buf304, (512, 768), (768, 1), 0), reinterpret_tensor(arg183_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf305)
    del arg183_1
    del arg184_1
    buf306 = buf275; del buf275  # reuse
    buf307 = buf274; del buf274  # reuse
    buf309 = reinterpret_tensor(buf304, (1, 512, 768), (393216, 768, 1), 0); del buf304  # reuse
    cpp_fused_add_native_layer_norm_62(c_void_p(buf305.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf309.data_ptr()))
    del arg185_1
    del arg186_1
    buf310 = reinterpret_tensor(buf272, (512, 3072), (3072, 1), 0); del buf272  # reuse
    # Source Nodes: [hidden_states_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg188_1, reinterpret_tensor(buf309, (512, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf310)
    del arg187_1
    del arg188_1
    buf311 = reinterpret_tensor(buf310, (1, 512, 3072), (1572864, 3072, 1), 0); del buf310  # reuse
    cpp_fused_gelu_63(c_void_p(buf311.data_ptr()))
    buf312 = buf305; del buf305  # reuse
    # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg190_1, reinterpret_tensor(buf311, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg189_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf312)
    del arg189_1
    del arg190_1
    buf313 = buf307; del buf307  # reuse
    buf314 = buf306; del buf306  # reuse
    buf316 = buf277; del buf277  # reuse
    cpp_fused_add_native_layer_norm_64(c_void_p(buf312.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf316.data_ptr()))
    del arg191_1
    del arg192_1
    del buf309
    buf317 = reinterpret_tensor(buf303, (512, 384), (384, 1), 0); del buf303  # reuse
    # Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg194_1, reinterpret_tensor(buf316, (512, 768), (768, 1), 0), reinterpret_tensor(arg193_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf317)
    del arg193_1
    del arg194_1
    buf318 = reinterpret_tensor(buf285, (512, 384), (384, 1), 0); del buf285  # reuse
    # Source Nodes: [mixed_key_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg196_1, reinterpret_tensor(buf316, (512, 768), (768, 1), 0), reinterpret_tensor(arg195_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf318)
    del arg195_1
    del arg196_1
    buf319 = reinterpret_tensor(buf296, (512, 384), (384, 1), 0); del buf296  # reuse
    # Source Nodes: [mixed_value_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg198_1, reinterpret_tensor(buf316, (512, 768), (768, 1), 0), reinterpret_tensor(arg197_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf319)
    del arg197_1
    del arg198_1
    buf320 = reinterpret_tensor(buf295, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf295  # reuse
    buf321 = reinterpret_tensor(buf318, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf318  # reuse
    buf322 = reinterpret_tensor(buf319, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf319  # reuse
    cpp_fused_65(c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf320.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf323 = aten._scaled_dot_product_flash_attention(buf320, buf321, buf322, scale=0.125)
    del buf320
    del buf321
    buf324 = buf323[0]
    del buf323
    buf331 = reinterpret_tensor(buf322, (512, 384), (384, 1), 0); del buf322  # reuse
    # Source Nodes: [conv_out_layer_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg204_1, reinterpret_tensor(buf316, (512, 768), (768, 1), 0), reinterpret_tensor(arg203_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf331)
    del arg203_1
    del arg204_1
    buf332 = reinterpret_tensor(buf312, (1, 768, 512), (393216, 512, 1), 0); del buf312  # reuse
    cpp_fused_convolution_66(c_void_p(buf316.data_ptr()), c_void_p(buf332.data_ptr()))
    # Source Nodes: [x_48], Original ATen: [aten.convolution]
    buf333 = extern_kernels.convolution(buf332, arg199_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf333, (1, 768, 512), (393216, 512, 1))
    del arg199_1
    # Source Nodes: [x_49], Original ATen: [aten.convolution]
    buf334 = extern_kernels.convolution(buf333, arg200_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf334, (1, 384, 512), (196608, 512, 1))
    del arg200_1
    buf335 = reinterpret_tensor(buf317, (1, 512, 384), (196608, 384, 1), 0); del buf317  # reuse
    cpp_fused_mul_67(c_void_p(buf335.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(arg8_1.data_ptr()))
    del arg8_1
    buf336 = reinterpret_tensor(buf302, (512, 54), (54, 1), 0); del buf302  # reuse
    # Source Nodes: [conv_kernel_layer_24], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf335, (512, 384), (384, 1), 0), reinterpret_tensor(arg201_1, (384, 54), (1, 384), 0), out=buf336)
    del arg201_1
    buf337 = buf300; del buf300  # reuse
    buf338 = reinterpret_tensor(buf336, (3072, 9, 1), (9, 1, 27648), 0); del buf336  # reuse
    buf339 = buf298; del buf298  # reuse
    buf340 = buf301; del buf301  # reuse
    buf341 = buf338; del buf338  # reuse
    cpp_fused__softmax_clone_68(c_void_p(buf341.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    del arg202_1
    buf342 = reinterpret_tensor(buf331, (3072, 64, 1), (64, 1, 1), 0); del buf331  # reuse
    # Source Nodes: [conv_kernel_layer_26, conv_out_layer_70], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf340, (3072, 64, 9), (576, 9, 1), 0), buf341, out=buf342)
    buf343 = reinterpret_tensor(buf333, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf333  # reuse
    cpp_fused_cat_69(c_void_p(buf324.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()))
    buf344 = reinterpret_tensor(buf332, (512, 768), (768, 1), 0); del buf332  # reuse
    # Source Nodes: [hidden_states_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg206_1, reinterpret_tensor(buf343, (512, 768), (768, 1), 0), reinterpret_tensor(arg205_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf344)
    del arg205_1
    del arg206_1
    buf345 = buf314; del buf314  # reuse
    buf346 = buf313; del buf313  # reuse
    buf348 = reinterpret_tensor(buf343, (1, 512, 768), (393216, 768, 1), 0); del buf343  # reuse
    cpp_fused_add_native_layer_norm_70(c_void_p(buf344.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf348.data_ptr()))
    del arg207_1
    del arg208_1
    buf349 = reinterpret_tensor(buf311, (512, 3072), (3072, 1), 0); del buf311  # reuse
    # Source Nodes: [hidden_states_76], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg210_1, reinterpret_tensor(buf348, (512, 768), (768, 1), 0), reinterpret_tensor(arg209_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf349)
    del arg209_1
    del arg210_1
    buf350 = reinterpret_tensor(buf349, (1, 512, 3072), (1572864, 3072, 1), 0); del buf349  # reuse
    cpp_fused_gelu_71(c_void_p(buf350.data_ptr()))
    buf351 = buf344; del buf344  # reuse
    # Source Nodes: [hidden_states_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg212_1, reinterpret_tensor(buf350, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg211_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf351)
    del arg211_1
    del arg212_1
    buf352 = buf346; del buf346  # reuse
    buf353 = buf345; del buf345  # reuse
    buf355 = buf316; del buf316  # reuse
    cpp_fused_add_native_layer_norm_72(c_void_p(buf351.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf355.data_ptr()))
    del arg213_1
    del arg214_1
    del buf348
    buf356 = reinterpret_tensor(buf342, (512, 384), (384, 1), 0); del buf342  # reuse
    # Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg216_1, reinterpret_tensor(buf355, (512, 768), (768, 1), 0), reinterpret_tensor(arg215_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf356)
    del arg215_1
    del arg216_1
    buf357 = reinterpret_tensor(buf324, (512, 384), (384, 1), 0); del buf324  # reuse
    # Source Nodes: [mixed_key_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg218_1, reinterpret_tensor(buf355, (512, 768), (768, 1), 0), reinterpret_tensor(arg217_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf357)
    del arg217_1
    del arg218_1
    buf358 = reinterpret_tensor(buf335, (512, 384), (384, 1), 0); del buf335  # reuse
    # Source Nodes: [mixed_value_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg220_1, reinterpret_tensor(buf355, (512, 768), (768, 1), 0), reinterpret_tensor(arg219_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf358)
    del arg219_1
    del arg220_1
    buf359 = reinterpret_tensor(buf334, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf334  # reuse
    buf360 = reinterpret_tensor(buf357, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf357  # reuse
    buf361 = reinterpret_tensor(buf358, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf358  # reuse
    cpp_fused_73(c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf359.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf362 = aten._scaled_dot_product_flash_attention(buf359, buf360, buf361, scale=0.125)
    del buf359
    del buf360
    buf363 = buf362[0]
    del buf362
    buf370 = reinterpret_tensor(buf361, (512, 384), (384, 1), 0); del buf361  # reuse
    # Source Nodes: [conv_out_layer_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg226_1, reinterpret_tensor(buf355, (512, 768), (768, 1), 0), reinterpret_tensor(arg225_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf370)
    del arg225_1
    del arg226_1
    buf371 = reinterpret_tensor(buf351, (1, 768, 512), (393216, 512, 1), 0); del buf351  # reuse
    cpp_fused_convolution_74(c_void_p(buf355.data_ptr()), c_void_p(buf371.data_ptr()))
    # Source Nodes: [x_54], Original ATen: [aten.convolution]
    buf372 = extern_kernels.convolution(buf371, arg221_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf372, (1, 768, 512), (393216, 512, 1))
    del arg221_1
    # Source Nodes: [x_55], Original ATen: [aten.convolution]
    buf373 = extern_kernels.convolution(buf372, arg222_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf373, (1, 384, 512), (196608, 512, 1))
    del arg222_1
    buf374 = reinterpret_tensor(buf356, (1, 512, 384), (196608, 384, 1), 0); del buf356  # reuse
    cpp_fused_mul_75(c_void_p(buf374.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg9_1
    buf375 = reinterpret_tensor(buf341, (512, 54), (54, 1), 0); del buf341  # reuse
    # Source Nodes: [conv_kernel_layer_27], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf374, (512, 384), (384, 1), 0), reinterpret_tensor(arg223_1, (384, 54), (1, 384), 0), out=buf375)
    del arg223_1
    buf376 = buf339; del buf339  # reuse
    buf377 = reinterpret_tensor(buf375, (3072, 9, 1), (9, 1, 27648), 0); del buf375  # reuse
    buf378 = buf337; del buf337  # reuse
    buf379 = buf340; del buf340  # reuse
    buf380 = buf377; del buf377  # reuse
    cpp_fused__softmax_clone_76(c_void_p(buf380.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()))
    del arg224_1
    buf381 = reinterpret_tensor(buf370, (3072, 64, 1), (64, 1, 1), 0); del buf370  # reuse
    # Source Nodes: [conv_kernel_layer_29, conv_out_layer_78], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf379, (3072, 64, 9), (576, 9, 1), 0), buf380, out=buf381)
    buf382 = reinterpret_tensor(buf372, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf372  # reuse
    cpp_fused_cat_77(c_void_p(buf363.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()))
    buf383 = reinterpret_tensor(buf371, (512, 768), (768, 1), 0); del buf371  # reuse
    # Source Nodes: [hidden_states_82], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg228_1, reinterpret_tensor(buf382, (512, 768), (768, 1), 0), reinterpret_tensor(arg227_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf383)
    del arg227_1
    del arg228_1
    buf384 = buf353; del buf353  # reuse
    buf385 = buf352; del buf352  # reuse
    buf387 = reinterpret_tensor(buf382, (1, 512, 768), (393216, 768, 1), 0); del buf382  # reuse
    cpp_fused_add_native_layer_norm_78(c_void_p(buf383.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf387.data_ptr()))
    del arg229_1
    del arg230_1
    buf388 = reinterpret_tensor(buf350, (512, 3072), (3072, 1), 0); del buf350  # reuse
    # Source Nodes: [hidden_states_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg232_1, reinterpret_tensor(buf387, (512, 768), (768, 1), 0), reinterpret_tensor(arg231_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf388)
    del arg231_1
    del arg232_1
    buf389 = reinterpret_tensor(buf388, (1, 512, 3072), (1572864, 3072, 1), 0); del buf388  # reuse
    cpp_fused_gelu_79(c_void_p(buf389.data_ptr()))
    buf390 = buf383; del buf383  # reuse
    # Source Nodes: [hidden_states_87], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg234_1, reinterpret_tensor(buf389, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg233_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf390)
    del arg233_1
    del arg234_1
    buf391 = buf385; del buf385  # reuse
    buf392 = buf384; del buf384  # reuse
    buf394 = buf355; del buf355  # reuse
    cpp_fused_add_native_layer_norm_80(c_void_p(buf390.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf394.data_ptr()))
    del arg235_1
    del arg236_1
    del buf387
    buf395 = reinterpret_tensor(buf381, (512, 384), (384, 1), 0); del buf381  # reuse
    # Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg238_1, reinterpret_tensor(buf394, (512, 768), (768, 1), 0), reinterpret_tensor(arg237_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf395)
    del arg237_1
    del arg238_1
    buf396 = reinterpret_tensor(buf363, (512, 384), (384, 1), 0); del buf363  # reuse
    # Source Nodes: [mixed_key_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg240_1, reinterpret_tensor(buf394, (512, 768), (768, 1), 0), reinterpret_tensor(arg239_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf396)
    del arg239_1
    del arg240_1
    buf397 = reinterpret_tensor(buf374, (512, 384), (384, 1), 0); del buf374  # reuse
    # Source Nodes: [mixed_value_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg242_1, reinterpret_tensor(buf394, (512, 768), (768, 1), 0), reinterpret_tensor(arg241_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf397)
    del arg241_1
    del arg242_1
    buf398 = reinterpret_tensor(buf373, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf373  # reuse
    buf399 = reinterpret_tensor(buf396, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf396  # reuse
    buf400 = reinterpret_tensor(buf397, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf397  # reuse
    cpp_fused_81(c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf398.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf401 = aten._scaled_dot_product_flash_attention(buf398, buf399, buf400, scale=0.125)
    del buf398
    del buf399
    buf402 = buf401[0]
    del buf401
    buf409 = reinterpret_tensor(buf400, (512, 384), (384, 1), 0); del buf400  # reuse
    # Source Nodes: [conv_out_layer_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg248_1, reinterpret_tensor(buf394, (512, 768), (768, 1), 0), reinterpret_tensor(arg247_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf409)
    del arg247_1
    del arg248_1
    buf410 = reinterpret_tensor(buf390, (1, 768, 512), (393216, 512, 1), 0); del buf390  # reuse
    cpp_fused_convolution_82(c_void_p(buf394.data_ptr()), c_void_p(buf410.data_ptr()))
    # Source Nodes: [x_60], Original ATen: [aten.convolution]
    buf411 = extern_kernels.convolution(buf410, arg243_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf411, (1, 768, 512), (393216, 512, 1))
    del arg243_1
    # Source Nodes: [x_61], Original ATen: [aten.convolution]
    buf412 = extern_kernels.convolution(buf411, arg244_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf412, (1, 384, 512), (196608, 512, 1))
    del arg244_1
    buf413 = reinterpret_tensor(buf395, (1, 512, 384), (196608, 384, 1), 0); del buf395  # reuse
    cpp_fused_mul_83(c_void_p(buf413.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(arg10_1.data_ptr()))
    del arg10_1
    buf414 = reinterpret_tensor(buf380, (512, 54), (54, 1), 0); del buf380  # reuse
    # Source Nodes: [conv_kernel_layer_30], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf413, (512, 384), (384, 1), 0), reinterpret_tensor(arg245_1, (384, 54), (1, 384), 0), out=buf414)
    del arg245_1
    buf415 = buf378; del buf378  # reuse
    buf416 = reinterpret_tensor(buf414, (3072, 9, 1), (9, 1, 27648), 0); del buf414  # reuse
    buf417 = buf376; del buf376  # reuse
    buf418 = buf379; del buf379  # reuse
    buf419 = buf416; del buf416  # reuse
    cpp_fused__softmax_clone_84(c_void_p(buf419.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()))
    del arg246_1
    buf420 = reinterpret_tensor(buf409, (3072, 64, 1), (64, 1, 1), 0); del buf409  # reuse
    # Source Nodes: [conv_kernel_layer_32, conv_out_layer_86], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf418, (3072, 64, 9), (576, 9, 1), 0), buf419, out=buf420)
    buf421 = reinterpret_tensor(buf411, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf411  # reuse
    cpp_fused_cat_85(c_void_p(buf402.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()))
    buf422 = reinterpret_tensor(buf410, (512, 768), (768, 1), 0); del buf410  # reuse
    # Source Nodes: [hidden_states_91], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg250_1, reinterpret_tensor(buf421, (512, 768), (768, 1), 0), reinterpret_tensor(arg249_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf422)
    del arg249_1
    del arg250_1
    buf423 = buf392; del buf392  # reuse
    buf424 = buf391; del buf391  # reuse
    buf426 = reinterpret_tensor(buf421, (1, 512, 768), (393216, 768, 1), 0); del buf421  # reuse
    cpp_fused_add_native_layer_norm_86(c_void_p(buf422.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf426.data_ptr()))
    del arg251_1
    del arg252_1
    buf427 = reinterpret_tensor(buf389, (512, 3072), (3072, 1), 0); del buf389  # reuse
    # Source Nodes: [hidden_states_94], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg254_1, reinterpret_tensor(buf426, (512, 768), (768, 1), 0), reinterpret_tensor(arg253_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf427)
    del arg253_1
    del arg254_1
    buf428 = reinterpret_tensor(buf427, (1, 512, 3072), (1572864, 3072, 1), 0); del buf427  # reuse
    cpp_fused_gelu_87(c_void_p(buf428.data_ptr()))
    buf429 = buf422; del buf422  # reuse
    # Source Nodes: [hidden_states_96], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg256_1, reinterpret_tensor(buf428, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg255_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf429)
    del arg255_1
    del arg256_1
    buf430 = buf424; del buf424  # reuse
    buf431 = buf423; del buf423  # reuse
    buf433 = buf394; del buf394  # reuse
    cpp_fused_add_native_layer_norm_88(c_void_p(buf429.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf433.data_ptr()))
    del arg257_1
    del arg258_1
    del buf426
    buf434 = reinterpret_tensor(buf420, (512, 384), (384, 1), 0); del buf420  # reuse
    # Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg260_1, reinterpret_tensor(buf433, (512, 768), (768, 1), 0), reinterpret_tensor(arg259_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf434)
    del arg259_1
    del arg260_1
    buf435 = reinterpret_tensor(buf402, (512, 384), (384, 1), 0); del buf402  # reuse
    # Source Nodes: [mixed_key_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg262_1, reinterpret_tensor(buf433, (512, 768), (768, 1), 0), reinterpret_tensor(arg261_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf435)
    del arg261_1
    del arg262_1
    buf436 = reinterpret_tensor(buf413, (512, 384), (384, 1), 0); del buf413  # reuse
    # Source Nodes: [mixed_value_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg264_1, reinterpret_tensor(buf433, (512, 768), (768, 1), 0), reinterpret_tensor(arg263_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf436)
    del arg263_1
    del arg264_1
    buf437 = reinterpret_tensor(buf412, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf412  # reuse
    buf438 = reinterpret_tensor(buf435, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf435  # reuse
    buf439 = reinterpret_tensor(buf436, (1, 6, 512, 64), (196608, 64, 384, 1), 0); del buf436  # reuse
    cpp_fused_89(c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf437.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf440 = aten._scaled_dot_product_flash_attention(buf437, buf438, buf439, scale=0.125)
    del buf437
    del buf438
    buf441 = buf440[0]
    del buf440
    buf448 = reinterpret_tensor(buf439, (512, 384), (384, 1), 0); del buf439  # reuse
    # Source Nodes: [conv_out_layer_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg270_1, reinterpret_tensor(buf433, (512, 768), (768, 1), 0), reinterpret_tensor(arg269_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf448)
    del arg269_1
    del arg270_1
    buf449 = reinterpret_tensor(buf429, (1, 768, 512), (393216, 512, 1), 0); del buf429  # reuse
    cpp_fused_convolution_90(c_void_p(buf433.data_ptr()), c_void_p(buf449.data_ptr()))
    # Source Nodes: [x_66], Original ATen: [aten.convolution]
    buf450 = extern_kernels.convolution(buf449, arg265_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf450, (1, 768, 512), (393216, 512, 1))
    del arg265_1
    # Source Nodes: [x_67], Original ATen: [aten.convolution]
    buf451 = extern_kernels.convolution(buf450, arg266_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf451, (1, 384, 512), (196608, 512, 1))
    del arg266_1
    buf452 = reinterpret_tensor(buf434, (1, 512, 384), (196608, 384, 1), 0); del buf434  # reuse
    cpp_fused_mul_91(c_void_p(buf452.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(arg11_1.data_ptr()))
    del arg11_1
    del buf451
    buf453 = reinterpret_tensor(buf419, (512, 54), (54, 1), 0); del buf419  # reuse
    # Source Nodes: [conv_kernel_layer_33], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf452, (512, 384), (384, 1), 0), reinterpret_tensor(arg267_1, (384, 54), (1, 384), 0), out=buf453)
    del arg267_1
    del buf452
    buf454 = buf417; del buf417  # reuse
    buf455 = reinterpret_tensor(buf453, (3072, 9, 1), (9, 1, 27648), 0); del buf453  # reuse
    buf456 = buf415; del buf415  # reuse
    buf457 = buf418; del buf418  # reuse
    buf458 = buf455; del buf455  # reuse
    cpp_fused__softmax_clone_92(c_void_p(buf458.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()))
    del arg268_1
    del buf454
    del buf456
    buf459 = reinterpret_tensor(buf448, (3072, 64, 1), (64, 1, 1), 0); del buf448  # reuse
    # Source Nodes: [conv_kernel_layer_35, conv_out_layer_94], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf457, (3072, 64, 9), (576, 9, 1), 0), buf458, out=buf459)
    del buf457
    del buf458
    buf460 = reinterpret_tensor(buf450, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf450  # reuse
    cpp_fused_cat_93(c_void_p(buf441.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()))
    del buf441
    del buf459
    buf461 = reinterpret_tensor(buf449, (512, 768), (768, 1), 0); del buf449  # reuse
    # Source Nodes: [hidden_states_100], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg272_1, reinterpret_tensor(buf460, (512, 768), (768, 1), 0), reinterpret_tensor(arg271_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf461)
    del arg271_1
    del arg272_1
    buf462 = buf431; del buf431  # reuse
    buf463 = buf430; del buf430  # reuse
    buf465 = reinterpret_tensor(buf460, (1, 512, 768), (393216, 768, 1), 0); del buf460  # reuse
    cpp_fused_add_native_layer_norm_94(c_void_p(buf461.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf465.data_ptr()))
    del arg273_1
    del arg274_1
    buf466 = reinterpret_tensor(buf428, (512, 3072), (3072, 1), 0); del buf428  # reuse
    # Source Nodes: [hidden_states_103], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg276_1, reinterpret_tensor(buf465, (512, 768), (768, 1), 0), reinterpret_tensor(arg275_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf466)
    del arg275_1
    del arg276_1
    buf467 = reinterpret_tensor(buf466, (1, 512, 3072), (1572864, 3072, 1), 0); del buf466  # reuse
    cpp_fused_gelu_95(c_void_p(buf467.data_ptr()))
    buf468 = buf461; del buf461  # reuse
    # Source Nodes: [hidden_states_105], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg278_1, reinterpret_tensor(buf467, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg277_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf468)
    del arg277_1
    del arg278_1
    del buf467
    buf469 = buf463; del buf463  # reuse
    buf470 = buf462; del buf462  # reuse
    buf472 = buf433; del buf433  # reuse
    cpp_fused_add_native_layer_norm_96(c_void_p(buf468.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf472.data_ptr()))
    del arg279_1
    del arg280_1
    del buf465
    buf473 = buf468; del buf468  # reuse
    # Source Nodes: [hidden_states_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg282_1, reinterpret_tensor(buf472, (512, 768), (768, 1), 0), reinterpret_tensor(arg281_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf473)
    del arg281_1
    del arg282_1
    buf474 = buf470; del buf470  # reuse
    buf475 = buf469; del buf469  # reuse
    buf477 = buf472; del buf472  # reuse
    cpp_fused_gelu_native_layer_norm_97(c_void_p(buf473.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf477.data_ptr()))
    del arg283_1
    del arg284_1
    del buf473
    buf478 = empty((512, 30522), device='cpu', dtype=torch.float32)
    # Source Nodes: [prediction_scores_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg286_1, reinterpret_tensor(buf477, (512, 768), (768, 1), 0), reinterpret_tensor(arg285_1, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf478)
    del arg285_1
    del arg286_1
    del buf477
    buf479 = reinterpret_tensor(buf475, (512, 1), (1, 512), 0); del buf475  # reuse
    buf480 = reinterpret_tensor(buf474, (512, 1), (1, 512), 0); del buf474  # reuse
    buf481 = empty((), device='cpu', dtype=torch.float32)
    buf482 = empty((), device='cpu', dtype=torch.int64)
    buf483 = buf481; del buf481  # reuse
    cpp_fused__log_softmax_nll_loss_forward_98(c_void_p(buf483.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf482.data_ptr()))
    del arg290_1
    return (buf483, reinterpret_tensor(buf478, (1, 512, 30522), (15627264, 30522, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
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
    arg149_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((30522, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg288_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg289_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg290_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('YituTechConvBert', benchmark_compiled_module)
