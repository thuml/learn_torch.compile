
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


cpp_fused_convolution_0 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (256L*x1) + (768L*x0))];
                        out_ptr1[static_cast<long>(x1 + (3L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (768L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (150528L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp4));
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(in_ptr1 + static_cast<long>(x2 + (768L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (150528L*x0)), to_float_mask(tmp8));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp8));
                            auto tmp22 = decltype(tmp18)::blendv(tmp21, tmp18, tmp14);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp15);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp22);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp19 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (768L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (150528L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 - tmp17;
                        auto tmp20 = static_cast<float>(768.0);
                        auto tmp21 = tmp19 / tmp20;
                        auto tmp22 = static_cast<float>(1e-06);
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        auto tmp24 = 1 / std::sqrt(tmp23);
                        auto tmp25 = at::vec::Vectorized<float>(tmp24);
                        auto tmp26 = tmp18 * tmp25;
                        auto tmp28 = tmp26 * tmp27;
                        auto tmp30 = tmp28 + tmp29;
                        tmp30.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr4[static_cast<long>(x0)];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr5[static_cast<long>((-768L) + x0)];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(2304);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr6[static_cast<long>((-1536L) + x0)];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr3[static_cast<long>(x0)] = tmp22;
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.3535533905932738);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 732);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 732);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_mul_native_layer_norm_5 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (768L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (150528L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp18 = tmp16 * tmp17;
                            auto tmp19 = tmp15 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (768L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (150528L*x0)), to_float_mask(tmp8));
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp22)::blendv(tmp25, tmp22, tmp14);
                            auto tmp27 = tmp26 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp27);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                        auto tmp20 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (768L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (150528L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp18 = tmp16 * tmp17;
                        auto tmp19 = tmp15 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(768.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_mul_native_layer_norm_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (768L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (150528L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp18 = tmp16 * tmp17;
                        auto tmp19 = tmp15 + tmp18;
                        auto tmp22 = tmp20 * tmp21;
                        auto tmp23 = tmp19 + tmp22;
                        tmp23.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr7[static_cast<long>(x0)];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr8[static_cast<long>((-768L) + x0)];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(2304);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr9[static_cast<long>((-1536L) + x0)];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr3[static_cast<long>(x0)] = tmp22;
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.3535533905932738);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 732);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 732);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
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


cpp_fused_gelu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_mul_native_layer_norm_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp7 = tmp5 * tmp6;
                        auto tmp8 = tmp4 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr6[static_cast<long>(x0)];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr7[static_cast<long>((-768L) + x0)];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(2304);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr8[static_cast<long>((-1536L) + x0)];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr3[static_cast<long>(x0)] = tmp22;
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.3535533905932738);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 732);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 732);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
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


cpp_fused_gelu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_mul_native_layer_norm_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp7 = tmp5 * tmp6;
                        auto tmp8 = tmp4 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr6[static_cast<long>(x0)];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr7[static_cast<long>((-768L) + x0)];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(2304);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr8[static_cast<long>((-1536L) + x0)];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr3[static_cast<long>(x0)] = tmp22;
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.3535533905932738);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 732);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 732);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
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


cpp_fused_gelu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_mul_native_layer_norm_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp7 = tmp5 * tmp6;
                        auto tmp8 = tmp4 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr6[static_cast<long>(x0)];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr7[static_cast<long>((-768L) + x0)];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(2304);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr8[static_cast<long>((-1536L) + x0)];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr3[static_cast<long>(x0)] = tmp22;
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.3535533905932738);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 732);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 732);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
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


cpp_fused_gelu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_mul_native_layer_norm_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp7 = tmp5 * tmp6;
                        auto tmp8 = tmp4 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr6[static_cast<long>(x0)];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr7[static_cast<long>((-768L) + x0)];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(2304);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr8[static_cast<long>((-1536L) + x0)];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr3[static_cast<long>(x0)] = tmp22;
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.3535533905932738);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 732);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 732);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
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


cpp_fused_gelu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_mul_native_layer_norm_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp7 = tmp5 * tmp6;
                        auto tmp8 = tmp4 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr6[static_cast<long>(x0)];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr7[static_cast<long>((-768L) + x0)];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(2304);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr8[static_cast<long>((-1536L) + x0)];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr3[static_cast<long>(x0)] = tmp22;
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.3535533905932738);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 732);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 732);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
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


cpp_fused_gelu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_mul_native_layer_norm_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp7 = tmp5 * tmp6;
                        auto tmp8 = tmp4 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr6[static_cast<long>(x0)];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr7[static_cast<long>((-768L) + x0)];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(2304);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr8[static_cast<long>((-1536L) + x0)];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr3[static_cast<long>(x0)] = tmp22;
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.3535533905932738);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 732);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 732);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
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


cpp_fused_gelu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_mul_native_layer_norm_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp7 = tmp5 * tmp6;
                        auto tmp8 = tmp4 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr6[static_cast<long>(x0)];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr7[static_cast<long>((-768L) + x0)];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(2304);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr8[static_cast<long>((-1536L) + x0)];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr3[static_cast<long>(x0)] = tmp22;
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.3535533905932738);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 732);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 732);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
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


cpp_fused_gelu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_mul_native_layer_norm_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp7 = tmp5 * tmp6;
                        auto tmp8 = tmp4 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr6[static_cast<long>(x0)];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr7[static_cast<long>((-768L) + x0)];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(2304);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr8[static_cast<long>((-1536L) + x0)];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr3[static_cast<long>(x0)] = tmp22;
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.3535533905932738);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 732);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 732);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
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


cpp_fused_gelu_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_mul_native_layer_norm_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp7 = tmp5 * tmp6;
                        auto tmp8 = tmp4 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr6[static_cast<long>(x0)];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr7[static_cast<long>((-768L) + x0)];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(2304);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr8[static_cast<long>((-1536L) + x0)];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr3[static_cast<long>(x0)] = tmp22;
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.3535533905932738);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 732);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 732);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
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


cpp_fused_gelu_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_mul_native_layer_norm_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp7 = tmp5 * tmp6;
                        auto tmp8 = tmp4 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr6[static_cast<long>(x0)];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr7[static_cast<long>((-768L) + x0)];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(2304);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr8[static_cast<long>((-1536L) + x0)];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr3[static_cast<long>(x0)] = tmp22;
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.3535533905932738);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 732);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (197L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 732);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 732L), "index out of bounds: 0 <= tmp4 < 732L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (12L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
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


cpp_fused_gelu_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_mean_native_layer_norm_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (768L*x2) + (151296L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(768L + x1 + (768L*x2) + (151296L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(768L + x1 + (768L*x2) + (151296L*x0)));
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 + tmp3;
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp8 = tmp4 + tmp7;
                            tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                            auto tmp1 = static_cast<float>(196.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                        out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = out_ptr1[static_cast<long>(x0)];
                        auto tmp7 = out_ptr2[static_cast<long>(x0)];
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp1 = static_cast<float>(196.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp8 = static_cast<float>(768.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp6 * tmp13;
                        auto tmp16 = tmp14 * tmp15;
                        auto tmp18 = tmp16 + tmp17;
                        tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg1_1, (768, ), (1, ))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (2304, 768), (768, 1))
    assert_size_stride(arg7_1, (732, 12), (12, 1))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (2304, 768), (768, 1))
    assert_size_stride(arg17_1, (732, 12), (12, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (2304, 768), (768, 1))
    assert_size_stride(arg27_1, (732, 12), (12, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (2304, 768), (768, 1))
    assert_size_stride(arg37_1, (732, 12), (12, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (2304, 768), (768, 1))
    assert_size_stride(arg47_1, (732, 12), (12, 1))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (2304, 768), (768, 1))
    assert_size_stride(arg57_1, (732, 12), (12, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (2304, 768), (768, 1))
    assert_size_stride(arg67_1, (732, 12), (12, 1))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (2304, 768), (768, 1))
    assert_size_stride(arg77_1, (732, 12), (12, 1))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (2304, 768), (768, 1))
    assert_size_stride(arg87_1, (732, 12), (12, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (2304, 768), (768, 1))
    assert_size_stride(arg97_1, (732, 12), (12, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (2304, 768), (768, 1))
    assert_size_stride(arg107_1, (732, 12), (12, 1))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (2304, 768), (768, 1))
    assert_size_stride(arg117_1, (732, 12), (12, 1))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, 768), (768, 1))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (3072, 768), (768, 1))
    assert_size_stride(arg128_1, (3072, ), (1, ))
    assert_size_stride(arg129_1, (768, 3072), (3072, 1))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, 768), (768, 1))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (3072, 768), (768, 1))
    assert_size_stride(arg134_1, (3072, ), (1, ))
    assert_size_stride(arg135_1, (768, 3072), (3072, 1))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, 768), (768, 1))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (3072, 768), (768, 1))
    assert_size_stride(arg140_1, (3072, ), (1, ))
    assert_size_stride(arg141_1, (768, 3072), (3072, 1))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, 768), (768, 1))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (3072, 768), (768, 1))
    assert_size_stride(arg146_1, (3072, ), (1, ))
    assert_size_stride(arg147_1, (768, 3072), (3072, 1))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, 768), (768, 1))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (3072, 768), (768, 1))
    assert_size_stride(arg152_1, (3072, ), (1, ))
    assert_size_stride(arg153_1, (768, 3072), (3072, 1))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, 768), (768, 1))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (3072, 768), (768, 1))
    assert_size_stride(arg158_1, (3072, ), (1, ))
    assert_size_stride(arg159_1, (768, 3072), (3072, 1))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, 768), (768, 1))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (3072, 768), (768, 1))
    assert_size_stride(arg164_1, (3072, ), (1, ))
    assert_size_stride(arg165_1, (768, 3072), (3072, 1))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, 768), (768, 1))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (3072, 768), (768, 1))
    assert_size_stride(arg170_1, (3072, ), (1, ))
    assert_size_stride(arg171_1, (768, 3072), (3072, 1))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, 768), (768, 1))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (3072, 768), (768, 1))
    assert_size_stride(arg176_1, (3072, ), (1, ))
    assert_size_stride(arg177_1, (768, 3072), (3072, 1))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, 768), (768, 1))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (3072, 768), (768, 1))
    assert_size_stride(arg182_1, (3072, ), (1, ))
    assert_size_stride(arg183_1, (768, 3072), (3072, 1))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, 768), (768, 1))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (3072, 768), (768, 1))
    assert_size_stride(arg188_1, (3072, ), (1, ))
    assert_size_stride(arg189_1, (768, 3072), (3072, 1))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, 768), (768, 1))
    assert_size_stride(arg192_1, (768, ), (1, ))
    assert_size_stride(arg193_1, (3072, 768), (768, 1))
    assert_size_stride(arg194_1, (3072, ), (1, ))
    assert_size_stride(arg195_1, (768, 3072), (3072, 1))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (1000, 768), (768, 1))
    assert_size_stride(arg198_1, (1000, ), (1, ))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (197, 197), (197, 1))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (197, 197), (197, 1))
    assert_size_stride(arg203_1, (768, ), (1, ))
    assert_size_stride(arg204_1, (197, 197), (197, 1))
    assert_size_stride(arg205_1, (768, ), (1, ))
    assert_size_stride(arg206_1, (197, 197), (197, 1))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (197, 197), (197, 1))
    assert_size_stride(arg209_1, (768, ), (1, ))
    assert_size_stride(arg210_1, (197, 197), (197, 1))
    assert_size_stride(arg211_1, (768, ), (1, ))
    assert_size_stride(arg212_1, (197, 197), (197, 1))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (197, 197), (197, 1))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (197, 197), (197, 1))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (197, 197), (197, 1))
    assert_size_stride(arg219_1, (768, ), (1, ))
    assert_size_stride(arg220_1, (197, 197), (197, 1))
    assert_size_stride(arg221_1, (768, ), (1, ))
    assert_size_stride(arg222_1, (197, 197), (197, 1))
    assert_size_stride(arg223_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((768, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg223_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg123_1
    del arg223_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg124_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 768, 14, 14), (150528, 1, 10752, 768))
    del arg124_1
    del buf0
    del buf1
    buf3 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    buf7 = empty((2304, ), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_1(c_void_p(arg0_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg199_1
    del arg2_1
    del arg3_1
    del arg4_1
    del arg5_1
    buf8 = empty((1576, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [cat_24, qkv], Original ATen: [aten.addmm, aten.cat]
    extern_kernels.addmm(buf7, reinterpret_tensor(buf6, (1576, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf8)
    del arg6_1
    buf9 = reinterpret_tensor(buf6, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf6  # reuse
    buf10 = empty((8, 12, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_2(c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()))
    buf11 = empty((96, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf9, (96, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf10, (96, 64, 197), (12608, 197, 1), 0), out=buf11)
    buf12 = empty_strided((8, 12, 197, 1), (2364, 197, 1, 18912), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf11, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf11  # reuse
    buf14 = empty_strided((8, 12, 197, 1), (2364, 197, 1, 18912), device='cpu', dtype=torch.float32)
    buf15 = buf13; del buf13  # reuse
    buf16 = buf9; del buf9  # reuse
    cpp_fused__softmax_add_clone_3(c_void_p(buf15.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf16.data_ptr()))
    del arg200_1
    del arg7_1
    buf17 = reinterpret_tensor(buf10, (96, 197, 64), (12608, 64, 1), 0); del buf10  # reuse
    # Source Nodes: [x_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf15, (96, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf16, (96, 197, 64), (12608, 64, 1), 0), out=buf17)
    buf18 = reinterpret_tensor(buf16, (8, 197, 12, 64), (151296, 768, 64, 1), 0); del buf16  # reuse
    cpp_fused_clone_4(c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    buf19 = reinterpret_tensor(buf17, (1576, 768), (768, 1), 0); del buf17  # reuse
    # Source Nodes: [x_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg126_1, reinterpret_tensor(buf18, (1576, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf19)
    del arg125_1
    del arg126_1
    buf20 = buf4; del buf4  # reuse
    buf21 = buf3; del buf3  # reuse
    buf23 = reinterpret_tensor(buf18, (8, 197, 768), (151296, 768, 1), 0); del buf18  # reuse
    cpp_fused_add_cat_mul_native_layer_norm_5(c_void_p(arg0_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()))
    del arg10_1
    del arg9_1
    buf24 = empty((1576, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg128_1, reinterpret_tensor(buf23, (1576, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf24)
    del arg127_1
    del arg128_1
    buf25 = reinterpret_tensor(buf24, (8, 197, 3072), (605184, 3072, 1), 0); del buf24  # reuse
    cpp_fused_gelu_6(c_void_p(buf25.data_ptr()))
    buf26 = reinterpret_tensor(buf23, (1576, 768), (768, 1), 0); del buf23  # reuse
    # Source Nodes: [x_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg130_1, reinterpret_tensor(buf25, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg129_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf26)
    del arg129_1
    del arg130_1
    buf27 = reinterpret_tensor(buf26, (8, 197, 768), (151296, 768, 1), 0); del buf26  # reuse
    buf28 = buf21; del buf21  # reuse
    buf29 = buf20; del buf20  # reuse
    buf31 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    buf32 = buf7; del buf7  # reuse
    cpp_fused_add_cat_mul_native_layer_norm_7(c_void_p(buf27.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()))
    del arg0_1
    del arg12_1
    del arg13_1
    del arg14_1
    del arg15_1
    del arg1_1
    del arg201_1
    del arg8_1
    del buf2
    buf33 = buf8; del buf8  # reuse
    # Source Nodes: [cat_23, qkv_2], Original ATen: [aten.addmm, aten.cat]
    extern_kernels.addmm(buf32, reinterpret_tensor(buf31, (1576, 768), (768, 1), 0), reinterpret_tensor(arg16_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf33)
    del arg16_1
    buf34 = reinterpret_tensor(buf31, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf31  # reuse
    buf35 = reinterpret_tensor(buf19, (8, 12, 64, 197), (151296, 12608, 197, 1), 0); del buf19  # reuse
    cpp_fused_clone_mul_8(c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()))
    buf36 = reinterpret_tensor(buf15, (96, 197, 197), (38809, 197, 1), 0); del buf15  # reuse
    # Source Nodes: [x_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf34, (96, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf35, (96, 64, 197), (12608, 197, 1), 0), out=buf36)
    buf37 = buf14; del buf14  # reuse
    buf38 = reinterpret_tensor(buf36, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf36  # reuse
    buf39 = buf12; del buf12  # reuse
    buf40 = buf38; del buf38  # reuse
    buf41 = reinterpret_tensor(buf35, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf35  # reuse
    cpp_fused__softmax_add_clone_9(c_void_p(buf40.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf41.data_ptr()))
    del arg17_1
    del arg202_1
    buf42 = reinterpret_tensor(buf34, (96, 197, 64), (12608, 64, 1), 0); del buf34  # reuse
    # Source Nodes: [x_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf40, (96, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf41, (96, 197, 64), (12608, 64, 1), 0), out=buf42)
    buf43 = reinterpret_tensor(buf41, (8, 197, 12, 64), (151296, 768, 64, 1), 0); del buf41  # reuse
    cpp_fused_clone_10(c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    buf44 = reinterpret_tensor(buf42, (1576, 768), (768, 1), 0); del buf42  # reuse
    # Source Nodes: [x_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg132_1, reinterpret_tensor(buf43, (1576, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf44)
    del arg131_1
    del arg132_1
    buf45 = buf29; del buf29  # reuse
    buf46 = buf28; del buf28  # reuse
    buf48 = reinterpret_tensor(buf43, (8, 197, 768), (151296, 768, 1), 0); del buf43  # reuse
    cpp_fused_add_mul_native_layer_norm_11(c_void_p(buf27.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()))
    del arg19_1
    del arg20_1
    buf49 = reinterpret_tensor(buf25, (1576, 3072), (3072, 1), 0); del buf25  # reuse
    # Source Nodes: [x_28], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg134_1, reinterpret_tensor(buf48, (1576, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf49)
    del arg133_1
    del arg134_1
    buf50 = reinterpret_tensor(buf49, (8, 197, 3072), (605184, 3072, 1), 0); del buf49  # reuse
    cpp_fused_gelu_12(c_void_p(buf50.data_ptr()))
    buf51 = reinterpret_tensor(buf48, (1576, 768), (768, 1), 0); del buf48  # reuse
    # Source Nodes: [x_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg136_1, reinterpret_tensor(buf50, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg135_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf51)
    del arg135_1
    del arg136_1
    buf52 = reinterpret_tensor(buf51, (8, 197, 768), (151296, 768, 1), 0); del buf51  # reuse
    buf53 = buf46; del buf46  # reuse
    buf54 = buf45; del buf45  # reuse
    buf56 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    buf57 = buf32; del buf32  # reuse
    cpp_fused_add_cat_mul_native_layer_norm_13(c_void_p(buf52.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg11_1
    del arg18_1
    del arg203_1
    del arg22_1
    del arg23_1
    del arg24_1
    del arg25_1
    buf58 = buf33; del buf33  # reuse
    # Source Nodes: [cat_22, qkv_4], Original ATen: [aten.addmm, aten.cat]
    extern_kernels.addmm(buf57, reinterpret_tensor(buf56, (1576, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf58)
    del arg26_1
    buf59 = reinterpret_tensor(buf56, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf56  # reuse
    buf60 = reinterpret_tensor(buf44, (8, 12, 64, 197), (151296, 12608, 197, 1), 0); del buf44  # reuse
    cpp_fused_clone_mul_14(c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    buf61 = reinterpret_tensor(buf40, (96, 197, 197), (38809, 197, 1), 0); del buf40  # reuse
    # Source Nodes: [x_37], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf59, (96, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf60, (96, 64, 197), (12608, 197, 1), 0), out=buf61)
    buf62 = buf39; del buf39  # reuse
    buf63 = reinterpret_tensor(buf61, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf61  # reuse
    buf64 = buf37; del buf37  # reuse
    buf65 = buf63; del buf63  # reuse
    buf66 = reinterpret_tensor(buf60, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf60  # reuse
    cpp_fused__softmax_add_clone_15(c_void_p(buf65.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()))
    del arg204_1
    del arg27_1
    buf67 = reinterpret_tensor(buf59, (96, 197, 64), (12608, 64, 1), 0); del buf59  # reuse
    # Source Nodes: [x_37], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf65, (96, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf66, (96, 197, 64), (12608, 64, 1), 0), out=buf67)
    buf68 = reinterpret_tensor(buf66, (8, 197, 12, 64), (151296, 768, 64, 1), 0); del buf66  # reuse
    cpp_fused_clone_16(c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    buf69 = reinterpret_tensor(buf67, (1576, 768), (768, 1), 0); del buf67  # reuse
    # Source Nodes: [x_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf68, (1576, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf69)
    del arg137_1
    del arg138_1
    buf70 = buf54; del buf54  # reuse
    buf71 = buf53; del buf53  # reuse
    buf73 = reinterpret_tensor(buf68, (8, 197, 768), (151296, 768, 1), 0); del buf68  # reuse
    cpp_fused_add_mul_native_layer_norm_17(c_void_p(buf52.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf73.data_ptr()))
    del arg29_1
    del arg30_1
    buf74 = reinterpret_tensor(buf50, (1576, 3072), (3072, 1), 0); del buf50  # reuse
    # Source Nodes: [x_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg140_1, reinterpret_tensor(buf73, (1576, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf74)
    del arg139_1
    del arg140_1
    buf75 = reinterpret_tensor(buf74, (8, 197, 3072), (605184, 3072, 1), 0); del buf74  # reuse
    cpp_fused_gelu_18(c_void_p(buf75.data_ptr()))
    buf76 = reinterpret_tensor(buf73, (1576, 768), (768, 1), 0); del buf73  # reuse
    # Source Nodes: [x_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg142_1, reinterpret_tensor(buf75, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg141_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf76)
    del arg141_1
    del arg142_1
    buf77 = reinterpret_tensor(buf76, (8, 197, 768), (151296, 768, 1), 0); del buf76  # reuse
    buf78 = buf71; del buf71  # reuse
    buf79 = buf70; del buf70  # reuse
    buf81 = buf27; del buf27  # reuse
    buf82 = buf57; del buf57  # reuse
    cpp_fused_add_cat_mul_native_layer_norm_19(c_void_p(buf77.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()))
    del arg205_1
    del arg21_1
    del arg28_1
    del arg32_1
    del arg33_1
    del arg34_1
    del arg35_1
    buf83 = buf58; del buf58  # reuse
    # Source Nodes: [cat_21, qkv_6], Original ATen: [aten.addmm, aten.cat]
    extern_kernels.addmm(buf82, reinterpret_tensor(buf81, (1576, 768), (768, 1), 0), reinterpret_tensor(arg36_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf83)
    del arg36_1
    buf84 = reinterpret_tensor(buf81, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf81  # reuse
    buf85 = reinterpret_tensor(buf69, (8, 12, 64, 197), (151296, 12608, 197, 1), 0); del buf69  # reuse
    cpp_fused_clone_mul_20(c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    buf86 = reinterpret_tensor(buf65, (96, 197, 197), (38809, 197, 1), 0); del buf65  # reuse
    # Source Nodes: [x_52], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf84, (96, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf85, (96, 64, 197), (12608, 197, 1), 0), out=buf86)
    buf87 = buf64; del buf64  # reuse
    buf88 = reinterpret_tensor(buf86, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf86  # reuse
    buf89 = buf62; del buf62  # reuse
    buf90 = buf88; del buf88  # reuse
    buf91 = reinterpret_tensor(buf85, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf85  # reuse
    cpp_fused__softmax_add_clone_21(c_void_p(buf90.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf91.data_ptr()))
    del arg206_1
    del arg37_1
    buf92 = reinterpret_tensor(buf84, (96, 197, 64), (12608, 64, 1), 0); del buf84  # reuse
    # Source Nodes: [x_52], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf90, (96, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf91, (96, 197, 64), (12608, 64, 1), 0), out=buf92)
    buf93 = reinterpret_tensor(buf91, (8, 197, 12, 64), (151296, 768, 64, 1), 0); del buf91  # reuse
    cpp_fused_clone_22(c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()))
    buf94 = reinterpret_tensor(buf92, (1576, 768), (768, 1), 0); del buf92  # reuse
    # Source Nodes: [x_54], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg144_1, reinterpret_tensor(buf93, (1576, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf94)
    del arg143_1
    del arg144_1
    buf95 = buf79; del buf79  # reuse
    buf96 = buf78; del buf78  # reuse
    buf98 = reinterpret_tensor(buf93, (8, 197, 768), (151296, 768, 1), 0); del buf93  # reuse
    cpp_fused_add_mul_native_layer_norm_23(c_void_p(buf77.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf98.data_ptr()))
    del arg39_1
    del arg40_1
    buf99 = reinterpret_tensor(buf75, (1576, 3072), (3072, 1), 0); del buf75  # reuse
    # Source Nodes: [x_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg146_1, reinterpret_tensor(buf98, (1576, 768), (768, 1), 0), reinterpret_tensor(arg145_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf99)
    del arg145_1
    del arg146_1
    buf100 = reinterpret_tensor(buf99, (8, 197, 3072), (605184, 3072, 1), 0); del buf99  # reuse
    cpp_fused_gelu_24(c_void_p(buf100.data_ptr()))
    buf101 = reinterpret_tensor(buf98, (1576, 768), (768, 1), 0); del buf98  # reuse
    # Source Nodes: [x_62], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg148_1, reinterpret_tensor(buf100, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg147_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf101)
    del arg147_1
    del arg148_1
    buf102 = reinterpret_tensor(buf101, (8, 197, 768), (151296, 768, 1), 0); del buf101  # reuse
    buf103 = buf96; del buf96  # reuse
    buf104 = buf95; del buf95  # reuse
    buf106 = buf52; del buf52  # reuse
    buf107 = buf82; del buf82  # reuse
    cpp_fused_add_cat_mul_native_layer_norm_25(c_void_p(buf102.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    del arg207_1
    del arg31_1
    del arg38_1
    del arg42_1
    del arg43_1
    del arg44_1
    del arg45_1
    buf108 = buf83; del buf83  # reuse
    # Source Nodes: [cat_20, qkv_8], Original ATen: [aten.addmm, aten.cat]
    extern_kernels.addmm(buf107, reinterpret_tensor(buf106, (1576, 768), (768, 1), 0), reinterpret_tensor(arg46_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf108)
    del arg46_1
    buf109 = reinterpret_tensor(buf106, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf106  # reuse
    buf110 = reinterpret_tensor(buf94, (8, 12, 64, 197), (151296, 12608, 197, 1), 0); del buf94  # reuse
    cpp_fused_clone_mul_26(c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()))
    buf111 = reinterpret_tensor(buf90, (96, 197, 197), (38809, 197, 1), 0); del buf90  # reuse
    # Source Nodes: [x_67], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf109, (96, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf110, (96, 64, 197), (12608, 197, 1), 0), out=buf111)
    buf112 = buf89; del buf89  # reuse
    buf113 = reinterpret_tensor(buf111, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf111  # reuse
    buf114 = buf87; del buf87  # reuse
    buf115 = buf113; del buf113  # reuse
    buf116 = reinterpret_tensor(buf110, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf110  # reuse
    cpp_fused__softmax_add_clone_27(c_void_p(buf115.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()))
    del arg208_1
    del arg47_1
    buf117 = reinterpret_tensor(buf109, (96, 197, 64), (12608, 64, 1), 0); del buf109  # reuse
    # Source Nodes: [x_67], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf115, (96, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf116, (96, 197, 64), (12608, 64, 1), 0), out=buf117)
    buf118 = reinterpret_tensor(buf116, (8, 197, 12, 64), (151296, 768, 64, 1), 0); del buf116  # reuse
    cpp_fused_clone_28(c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()))
    buf119 = reinterpret_tensor(buf117, (1576, 768), (768, 1), 0); del buf117  # reuse
    # Source Nodes: [x_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg150_1, reinterpret_tensor(buf118, (1576, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf119)
    del arg149_1
    del arg150_1
    buf120 = buf104; del buf104  # reuse
    buf121 = buf103; del buf103  # reuse
    buf123 = reinterpret_tensor(buf118, (8, 197, 768), (151296, 768, 1), 0); del buf118  # reuse
    cpp_fused_add_mul_native_layer_norm_29(c_void_p(buf102.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf123.data_ptr()))
    del arg49_1
    del arg50_1
    buf124 = reinterpret_tensor(buf100, (1576, 3072), (3072, 1), 0); del buf100  # reuse
    # Source Nodes: [x_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg152_1, reinterpret_tensor(buf123, (1576, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf124)
    del arg151_1
    del arg152_1
    buf125 = reinterpret_tensor(buf124, (8, 197, 3072), (605184, 3072, 1), 0); del buf124  # reuse
    cpp_fused_gelu_30(c_void_p(buf125.data_ptr()))
    buf126 = reinterpret_tensor(buf123, (1576, 768), (768, 1), 0); del buf123  # reuse
    # Source Nodes: [x_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg154_1, reinterpret_tensor(buf125, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg153_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf126)
    del arg153_1
    del arg154_1
    buf127 = reinterpret_tensor(buf126, (8, 197, 768), (151296, 768, 1), 0); del buf126  # reuse
    buf128 = buf121; del buf121  # reuse
    buf129 = buf120; del buf120  # reuse
    buf131 = buf77; del buf77  # reuse
    buf132 = buf107; del buf107  # reuse
    cpp_fused_add_cat_mul_native_layer_norm_31(c_void_p(buf127.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del arg209_1
    del arg41_1
    del arg48_1
    del arg52_1
    del arg53_1
    del arg54_1
    del arg55_1
    buf133 = buf108; del buf108  # reuse
    # Source Nodes: [cat_19, qkv_10], Original ATen: [aten.addmm, aten.cat]
    extern_kernels.addmm(buf132, reinterpret_tensor(buf131, (1576, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf133)
    del arg56_1
    buf134 = reinterpret_tensor(buf131, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf131  # reuse
    buf135 = reinterpret_tensor(buf119, (8, 12, 64, 197), (151296, 12608, 197, 1), 0); del buf119  # reuse
    cpp_fused_clone_mul_32(c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    buf136 = reinterpret_tensor(buf115, (96, 197, 197), (38809, 197, 1), 0); del buf115  # reuse
    # Source Nodes: [x_82], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf134, (96, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf135, (96, 64, 197), (12608, 197, 1), 0), out=buf136)
    buf137 = buf114; del buf114  # reuse
    buf138 = reinterpret_tensor(buf136, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf136  # reuse
    buf139 = buf112; del buf112  # reuse
    buf140 = buf138; del buf138  # reuse
    buf141 = reinterpret_tensor(buf135, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf135  # reuse
    cpp_fused__softmax_add_clone_33(c_void_p(buf140.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()))
    del arg210_1
    del arg57_1
    buf142 = reinterpret_tensor(buf134, (96, 197, 64), (12608, 64, 1), 0); del buf134  # reuse
    # Source Nodes: [x_82], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf140, (96, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf141, (96, 197, 64), (12608, 64, 1), 0), out=buf142)
    buf143 = reinterpret_tensor(buf141, (8, 197, 12, 64), (151296, 768, 64, 1), 0); del buf141  # reuse
    cpp_fused_clone_34(c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    buf144 = reinterpret_tensor(buf142, (1576, 768), (768, 1), 0); del buf142  # reuse
    # Source Nodes: [x_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg156_1, reinterpret_tensor(buf143, (1576, 768), (768, 1), 0), reinterpret_tensor(arg155_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf144)
    del arg155_1
    del arg156_1
    buf145 = buf129; del buf129  # reuse
    buf146 = buf128; del buf128  # reuse
    buf148 = reinterpret_tensor(buf143, (8, 197, 768), (151296, 768, 1), 0); del buf143  # reuse
    cpp_fused_add_mul_native_layer_norm_35(c_void_p(buf127.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf148.data_ptr()))
    del arg59_1
    del arg60_1
    buf149 = reinterpret_tensor(buf125, (1576, 3072), (3072, 1), 0); del buf125  # reuse
    # Source Nodes: [x_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg158_1, reinterpret_tensor(buf148, (1576, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf149)
    del arg157_1
    del arg158_1
    buf150 = reinterpret_tensor(buf149, (8, 197, 3072), (605184, 3072, 1), 0); del buf149  # reuse
    cpp_fused_gelu_36(c_void_p(buf150.data_ptr()))
    buf151 = reinterpret_tensor(buf148, (1576, 768), (768, 1), 0); del buf148  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg160_1, reinterpret_tensor(buf150, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg159_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf151)
    del arg159_1
    del arg160_1
    buf152 = reinterpret_tensor(buf151, (8, 197, 768), (151296, 768, 1), 0); del buf151  # reuse
    buf153 = buf146; del buf146  # reuse
    buf154 = buf145; del buf145  # reuse
    buf156 = buf102; del buf102  # reuse
    buf157 = buf132; del buf132  # reuse
    cpp_fused_add_cat_mul_native_layer_norm_37(c_void_p(buf152.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()))
    del arg211_1
    del arg51_1
    del arg58_1
    del arg62_1
    del arg63_1
    del arg64_1
    del arg65_1
    buf158 = buf133; del buf133  # reuse
    # Source Nodes: [cat_18, qkv_12], Original ATen: [aten.addmm, aten.cat]
    extern_kernels.addmm(buf157, reinterpret_tensor(buf156, (1576, 768), (768, 1), 0), reinterpret_tensor(arg66_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf158)
    del arg66_1
    buf159 = reinterpret_tensor(buf156, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf156  # reuse
    buf160 = reinterpret_tensor(buf144, (8, 12, 64, 197), (151296, 12608, 197, 1), 0); del buf144  # reuse
    cpp_fused_clone_mul_38(c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    buf161 = reinterpret_tensor(buf140, (96, 197, 197), (38809, 197, 1), 0); del buf140  # reuse
    # Source Nodes: [x_97], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf159, (96, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf160, (96, 64, 197), (12608, 197, 1), 0), out=buf161)
    buf162 = buf139; del buf139  # reuse
    buf163 = reinterpret_tensor(buf161, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf161  # reuse
    buf164 = buf137; del buf137  # reuse
    buf165 = buf163; del buf163  # reuse
    buf166 = reinterpret_tensor(buf160, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf160  # reuse
    cpp_fused__softmax_add_clone_39(c_void_p(buf165.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf166.data_ptr()))
    del arg212_1
    del arg67_1
    buf167 = reinterpret_tensor(buf159, (96, 197, 64), (12608, 64, 1), 0); del buf159  # reuse
    # Source Nodes: [x_97], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf165, (96, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf166, (96, 197, 64), (12608, 64, 1), 0), out=buf167)
    buf168 = reinterpret_tensor(buf166, (8, 197, 12, 64), (151296, 768, 64, 1), 0); del buf166  # reuse
    cpp_fused_clone_40(c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()))
    buf169 = reinterpret_tensor(buf167, (1576, 768), (768, 1), 0); del buf167  # reuse
    # Source Nodes: [x_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg162_1, reinterpret_tensor(buf168, (1576, 768), (768, 1), 0), reinterpret_tensor(arg161_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf169)
    del arg161_1
    del arg162_1
    buf170 = buf154; del buf154  # reuse
    buf171 = buf153; del buf153  # reuse
    buf173 = reinterpret_tensor(buf168, (8, 197, 768), (151296, 768, 1), 0); del buf168  # reuse
    cpp_fused_add_mul_native_layer_norm_41(c_void_p(buf152.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()))
    del arg69_1
    del arg70_1
    buf174 = reinterpret_tensor(buf150, (1576, 3072), (3072, 1), 0); del buf150  # reuse
    # Source Nodes: [x_103], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg164_1, reinterpret_tensor(buf173, (1576, 768), (768, 1), 0), reinterpret_tensor(arg163_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf174)
    del arg163_1
    del arg164_1
    buf175 = reinterpret_tensor(buf174, (8, 197, 3072), (605184, 3072, 1), 0); del buf174  # reuse
    cpp_fused_gelu_42(c_void_p(buf175.data_ptr()))
    buf176 = reinterpret_tensor(buf173, (1576, 768), (768, 1), 0); del buf173  # reuse
    # Source Nodes: [x_107], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg166_1, reinterpret_tensor(buf175, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg165_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf176)
    del arg165_1
    del arg166_1
    buf177 = reinterpret_tensor(buf176, (8, 197, 768), (151296, 768, 1), 0); del buf176  # reuse
    buf178 = buf171; del buf171  # reuse
    buf179 = buf170; del buf170  # reuse
    buf181 = buf127; del buf127  # reuse
    buf182 = buf157; del buf157  # reuse
    cpp_fused_add_cat_mul_native_layer_norm_43(c_void_p(buf177.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()))
    del arg213_1
    del arg61_1
    del arg68_1
    del arg72_1
    del arg73_1
    del arg74_1
    del arg75_1
    buf183 = buf158; del buf158  # reuse
    # Source Nodes: [cat_17, qkv_14], Original ATen: [aten.addmm, aten.cat]
    extern_kernels.addmm(buf182, reinterpret_tensor(buf181, (1576, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf183)
    del arg76_1
    buf184 = reinterpret_tensor(buf181, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf181  # reuse
    buf185 = reinterpret_tensor(buf169, (8, 12, 64, 197), (151296, 12608, 197, 1), 0); del buf169  # reuse
    cpp_fused_clone_mul_44(c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    buf186 = reinterpret_tensor(buf165, (96, 197, 197), (38809, 197, 1), 0); del buf165  # reuse
    # Source Nodes: [x_112], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf184, (96, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf185, (96, 64, 197), (12608, 197, 1), 0), out=buf186)
    buf187 = buf164; del buf164  # reuse
    buf188 = reinterpret_tensor(buf186, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf186  # reuse
    buf189 = buf162; del buf162  # reuse
    buf190 = buf188; del buf188  # reuse
    buf191 = reinterpret_tensor(buf185, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf185  # reuse
    cpp_fused__softmax_add_clone_45(c_void_p(buf190.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()))
    del arg214_1
    del arg77_1
    buf192 = reinterpret_tensor(buf184, (96, 197, 64), (12608, 64, 1), 0); del buf184  # reuse
    # Source Nodes: [x_112], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf190, (96, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf191, (96, 197, 64), (12608, 64, 1), 0), out=buf192)
    buf193 = reinterpret_tensor(buf191, (8, 197, 12, 64), (151296, 768, 64, 1), 0); del buf191  # reuse
    cpp_fused_clone_46(c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    buf194 = reinterpret_tensor(buf192, (1576, 768), (768, 1), 0); del buf192  # reuse
    # Source Nodes: [x_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg168_1, reinterpret_tensor(buf193, (1576, 768), (768, 1), 0), reinterpret_tensor(arg167_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf194)
    del arg167_1
    del arg168_1
    buf195 = buf179; del buf179  # reuse
    buf196 = buf178; del buf178  # reuse
    buf198 = reinterpret_tensor(buf193, (8, 197, 768), (151296, 768, 1), 0); del buf193  # reuse
    cpp_fused_add_mul_native_layer_norm_47(c_void_p(buf177.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()))
    del arg79_1
    del arg80_1
    buf199 = reinterpret_tensor(buf175, (1576, 3072), (3072, 1), 0); del buf175  # reuse
    # Source Nodes: [x_118], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg170_1, reinterpret_tensor(buf198, (1576, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf199)
    del arg169_1
    del arg170_1
    buf200 = reinterpret_tensor(buf199, (8, 197, 3072), (605184, 3072, 1), 0); del buf199  # reuse
    cpp_fused_gelu_48(c_void_p(buf200.data_ptr()))
    buf201 = reinterpret_tensor(buf198, (1576, 768), (768, 1), 0); del buf198  # reuse
    # Source Nodes: [x_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg172_1, reinterpret_tensor(buf200, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg171_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf201)
    del arg171_1
    del arg172_1
    buf202 = reinterpret_tensor(buf201, (8, 197, 768), (151296, 768, 1), 0); del buf201  # reuse
    buf203 = buf196; del buf196  # reuse
    buf204 = buf195; del buf195  # reuse
    buf206 = buf152; del buf152  # reuse
    buf207 = buf182; del buf182  # reuse
    cpp_fused_add_cat_mul_native_layer_norm_49(c_void_p(buf202.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()))
    del arg215_1
    del arg71_1
    del arg78_1
    del arg82_1
    del arg83_1
    del arg84_1
    del arg85_1
    buf208 = buf183; del buf183  # reuse
    # Source Nodes: [cat_16, qkv_16], Original ATen: [aten.addmm, aten.cat]
    extern_kernels.addmm(buf207, reinterpret_tensor(buf206, (1576, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf208)
    del arg86_1
    buf209 = reinterpret_tensor(buf206, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf206  # reuse
    buf210 = reinterpret_tensor(buf194, (8, 12, 64, 197), (151296, 12608, 197, 1), 0); del buf194  # reuse
    cpp_fused_clone_mul_50(c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()))
    buf211 = reinterpret_tensor(buf190, (96, 197, 197), (38809, 197, 1), 0); del buf190  # reuse
    # Source Nodes: [x_127], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf209, (96, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf210, (96, 64, 197), (12608, 197, 1), 0), out=buf211)
    buf212 = buf189; del buf189  # reuse
    buf213 = reinterpret_tensor(buf211, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf211  # reuse
    buf214 = buf187; del buf187  # reuse
    buf215 = buf213; del buf213  # reuse
    buf216 = reinterpret_tensor(buf210, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf210  # reuse
    cpp_fused__softmax_add_clone_51(c_void_p(buf215.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()))
    del arg216_1
    del arg87_1
    buf217 = reinterpret_tensor(buf209, (96, 197, 64), (12608, 64, 1), 0); del buf209  # reuse
    # Source Nodes: [x_127], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf215, (96, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf216, (96, 197, 64), (12608, 64, 1), 0), out=buf217)
    buf218 = reinterpret_tensor(buf216, (8, 197, 12, 64), (151296, 768, 64, 1), 0); del buf216  # reuse
    cpp_fused_clone_52(c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()))
    buf219 = reinterpret_tensor(buf217, (1576, 768), (768, 1), 0); del buf217  # reuse
    # Source Nodes: [x_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg174_1, reinterpret_tensor(buf218, (1576, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf219)
    del arg173_1
    del arg174_1
    buf220 = buf204; del buf204  # reuse
    buf221 = buf203; del buf203  # reuse
    buf223 = reinterpret_tensor(buf218, (8, 197, 768), (151296, 768, 1), 0); del buf218  # reuse
    cpp_fused_add_mul_native_layer_norm_53(c_void_p(buf202.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf223.data_ptr()))
    del arg89_1
    del arg90_1
    buf224 = reinterpret_tensor(buf200, (1576, 3072), (3072, 1), 0); del buf200  # reuse
    # Source Nodes: [x_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg176_1, reinterpret_tensor(buf223, (1576, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf224)
    del arg175_1
    del arg176_1
    buf225 = reinterpret_tensor(buf224, (8, 197, 3072), (605184, 3072, 1), 0); del buf224  # reuse
    cpp_fused_gelu_54(c_void_p(buf225.data_ptr()))
    buf226 = reinterpret_tensor(buf223, (1576, 768), (768, 1), 0); del buf223  # reuse
    # Source Nodes: [x_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg178_1, reinterpret_tensor(buf225, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg177_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf226)
    del arg177_1
    del arg178_1
    buf227 = reinterpret_tensor(buf226, (8, 197, 768), (151296, 768, 1), 0); del buf226  # reuse
    buf228 = buf221; del buf221  # reuse
    buf229 = buf220; del buf220  # reuse
    buf231 = buf177; del buf177  # reuse
    buf232 = buf207; del buf207  # reuse
    cpp_fused_add_cat_mul_native_layer_norm_55(c_void_p(buf227.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    del arg217_1
    del arg81_1
    del arg88_1
    del arg92_1
    del arg93_1
    del arg94_1
    del arg95_1
    buf233 = buf208; del buf208  # reuse
    # Source Nodes: [cat_15, qkv_18], Original ATen: [aten.addmm, aten.cat]
    extern_kernels.addmm(buf232, reinterpret_tensor(buf231, (1576, 768), (768, 1), 0), reinterpret_tensor(arg96_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf233)
    del arg96_1
    buf234 = reinterpret_tensor(buf231, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf231  # reuse
    buf235 = reinterpret_tensor(buf219, (8, 12, 64, 197), (151296, 12608, 197, 1), 0); del buf219  # reuse
    cpp_fused_clone_mul_56(c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    buf236 = reinterpret_tensor(buf215, (96, 197, 197), (38809, 197, 1), 0); del buf215  # reuse
    # Source Nodes: [x_142], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf234, (96, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf235, (96, 64, 197), (12608, 197, 1), 0), out=buf236)
    buf237 = buf214; del buf214  # reuse
    buf238 = reinterpret_tensor(buf236, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf236  # reuse
    buf239 = buf212; del buf212  # reuse
    buf240 = buf238; del buf238  # reuse
    buf241 = reinterpret_tensor(buf235, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf235  # reuse
    cpp_fused__softmax_add_clone_57(c_void_p(buf240.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf241.data_ptr()))
    del arg218_1
    del arg97_1
    buf242 = reinterpret_tensor(buf234, (96, 197, 64), (12608, 64, 1), 0); del buf234  # reuse
    # Source Nodes: [x_142], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf240, (96, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf241, (96, 197, 64), (12608, 64, 1), 0), out=buf242)
    buf243 = reinterpret_tensor(buf241, (8, 197, 12, 64), (151296, 768, 64, 1), 0); del buf241  # reuse
    cpp_fused_clone_58(c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()))
    buf244 = reinterpret_tensor(buf242, (1576, 768), (768, 1), 0); del buf242  # reuse
    # Source Nodes: [x_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg180_1, reinterpret_tensor(buf243, (1576, 768), (768, 1), 0), reinterpret_tensor(arg179_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf244)
    del arg179_1
    del arg180_1
    buf245 = buf229; del buf229  # reuse
    buf246 = buf228; del buf228  # reuse
    buf248 = reinterpret_tensor(buf243, (8, 197, 768), (151296, 768, 1), 0); del buf243  # reuse
    cpp_fused_add_mul_native_layer_norm_59(c_void_p(buf227.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf248.data_ptr()))
    del arg100_1
    del arg99_1
    buf249 = reinterpret_tensor(buf225, (1576, 3072), (3072, 1), 0); del buf225  # reuse
    # Source Nodes: [x_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg182_1, reinterpret_tensor(buf248, (1576, 768), (768, 1), 0), reinterpret_tensor(arg181_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf249)
    del arg181_1
    del arg182_1
    buf250 = reinterpret_tensor(buf249, (8, 197, 3072), (605184, 3072, 1), 0); del buf249  # reuse
    cpp_fused_gelu_60(c_void_p(buf250.data_ptr()))
    buf251 = reinterpret_tensor(buf248, (1576, 768), (768, 1), 0); del buf248  # reuse
    # Source Nodes: [x_152], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg184_1, reinterpret_tensor(buf250, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg183_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf251)
    del arg183_1
    del arg184_1
    buf252 = reinterpret_tensor(buf251, (8, 197, 768), (151296, 768, 1), 0); del buf251  # reuse
    buf253 = buf246; del buf246  # reuse
    buf254 = buf245; del buf245  # reuse
    buf256 = buf202; del buf202  # reuse
    buf257 = buf232; del buf232  # reuse
    cpp_fused_add_cat_mul_native_layer_norm_61(c_void_p(buf252.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()))
    del arg102_1
    del arg103_1
    del arg104_1
    del arg105_1
    del arg219_1
    del arg91_1
    del arg98_1
    buf258 = buf233; del buf233  # reuse
    # Source Nodes: [cat_14, qkv_20], Original ATen: [aten.addmm, aten.cat]
    extern_kernels.addmm(buf257, reinterpret_tensor(buf256, (1576, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf258)
    del arg106_1
    buf259 = reinterpret_tensor(buf256, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf256  # reuse
    buf260 = reinterpret_tensor(buf244, (8, 12, 64, 197), (151296, 12608, 197, 1), 0); del buf244  # reuse
    cpp_fused_clone_mul_62(c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    buf261 = reinterpret_tensor(buf240, (96, 197, 197), (38809, 197, 1), 0); del buf240  # reuse
    # Source Nodes: [x_157], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf259, (96, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf260, (96, 64, 197), (12608, 197, 1), 0), out=buf261)
    buf262 = buf239; del buf239  # reuse
    buf263 = reinterpret_tensor(buf261, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf261  # reuse
    buf264 = buf237; del buf237  # reuse
    buf265 = buf263; del buf263  # reuse
    buf266 = reinterpret_tensor(buf260, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf260  # reuse
    cpp_fused__softmax_add_clone_63(c_void_p(buf265.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()))
    del arg107_1
    del arg220_1
    buf267 = reinterpret_tensor(buf259, (96, 197, 64), (12608, 64, 1), 0); del buf259  # reuse
    # Source Nodes: [x_157], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf265, (96, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf266, (96, 197, 64), (12608, 64, 1), 0), out=buf267)
    buf268 = reinterpret_tensor(buf266, (8, 197, 12, 64), (151296, 768, 64, 1), 0); del buf266  # reuse
    cpp_fused_clone_64(c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    buf269 = reinterpret_tensor(buf267, (1576, 768), (768, 1), 0); del buf267  # reuse
    # Source Nodes: [x_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg186_1, reinterpret_tensor(buf268, (1576, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf269)
    del arg185_1
    del arg186_1
    buf270 = buf254; del buf254  # reuse
    buf271 = buf253; del buf253  # reuse
    buf273 = reinterpret_tensor(buf268, (8, 197, 768), (151296, 768, 1), 0); del buf268  # reuse
    cpp_fused_add_mul_native_layer_norm_65(c_void_p(buf252.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()))
    del arg109_1
    del arg110_1
    buf274 = reinterpret_tensor(buf250, (1576, 3072), (3072, 1), 0); del buf250  # reuse
    # Source Nodes: [x_163], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg188_1, reinterpret_tensor(buf273, (1576, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf274)
    del arg187_1
    del arg188_1
    buf275 = reinterpret_tensor(buf274, (8, 197, 3072), (605184, 3072, 1), 0); del buf274  # reuse
    cpp_fused_gelu_66(c_void_p(buf275.data_ptr()))
    buf276 = reinterpret_tensor(buf273, (1576, 768), (768, 1), 0); del buf273  # reuse
    # Source Nodes: [x_167], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg190_1, reinterpret_tensor(buf275, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg189_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf276)
    del arg189_1
    del arg190_1
    buf277 = reinterpret_tensor(buf276, (8, 197, 768), (151296, 768, 1), 0); del buf276  # reuse
    buf278 = buf271; del buf271  # reuse
    buf279 = buf270; del buf270  # reuse
    buf281 = buf227; del buf227  # reuse
    buf282 = buf257; del buf257  # reuse
    cpp_fused_add_cat_mul_native_layer_norm_67(c_void_p(buf277.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()))
    del arg101_1
    del arg108_1
    del arg112_1
    del arg113_1
    del arg114_1
    del arg115_1
    del arg221_1
    del buf252
    buf283 = buf258; del buf258  # reuse
    # Source Nodes: [cat_13, qkv_22], Original ATen: [aten.addmm, aten.cat]
    extern_kernels.addmm(buf282, reinterpret_tensor(buf281, (1576, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf283)
    del arg116_1
    del buf282
    buf284 = reinterpret_tensor(buf281, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf281  # reuse
    buf285 = reinterpret_tensor(buf269, (8, 12, 64, 197), (151296, 12608, 197, 1), 0); del buf269  # reuse
    cpp_fused_clone_mul_68(c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    buf286 = reinterpret_tensor(buf265, (96, 197, 197), (38809, 197, 1), 0); del buf265  # reuse
    # Source Nodes: [x_172], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf284, (96, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf285, (96, 64, 197), (12608, 197, 1), 0), out=buf286)
    buf287 = buf264; del buf264  # reuse
    buf288 = reinterpret_tensor(buf286, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf286  # reuse
    buf289 = buf262; del buf262  # reuse
    buf290 = buf288; del buf288  # reuse
    buf291 = reinterpret_tensor(buf285, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf285  # reuse
    cpp_fused__softmax_add_clone_69(c_void_p(buf290.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()))
    del arg117_1
    del arg222_1
    del buf283
    del buf287
    del buf289
    buf292 = reinterpret_tensor(buf284, (96, 197, 64), (12608, 64, 1), 0); del buf284  # reuse
    # Source Nodes: [x_172], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf290, (96, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf291, (96, 197, 64), (12608, 64, 1), 0), out=buf292)
    del buf290
    buf293 = reinterpret_tensor(buf291, (8, 197, 12, 64), (151296, 768, 64, 1), 0); del buf291  # reuse
    cpp_fused_clone_70(c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()))
    buf294 = reinterpret_tensor(buf292, (1576, 768), (768, 1), 0); del buf292  # reuse
    # Source Nodes: [x_174], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg192_1, reinterpret_tensor(buf293, (1576, 768), (768, 1), 0), reinterpret_tensor(arg191_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf294)
    del arg191_1
    del arg192_1
    buf295 = buf279; del buf279  # reuse
    buf296 = buf278; del buf278  # reuse
    buf298 = reinterpret_tensor(buf293, (8, 197, 768), (151296, 768, 1), 0); del buf293  # reuse
    cpp_fused_add_mul_native_layer_norm_71(c_void_p(buf277.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()))
    del arg119_1
    del arg120_1
    del buf295
    del buf296
    buf299 = reinterpret_tensor(buf275, (1576, 3072), (3072, 1), 0); del buf275  # reuse
    # Source Nodes: [x_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg194_1, reinterpret_tensor(buf298, (1576, 768), (768, 1), 0), reinterpret_tensor(arg193_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf299)
    del arg193_1
    del arg194_1
    buf300 = reinterpret_tensor(buf299, (8, 197, 3072), (605184, 3072, 1), 0); del buf299  # reuse
    cpp_fused_gelu_72(c_void_p(buf300.data_ptr()))
    buf301 = reinterpret_tensor(buf298, (1576, 768), (768, 1), 0); del buf298  # reuse
    # Source Nodes: [x_182], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg196_1, reinterpret_tensor(buf300, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg195_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf301)
    del arg195_1
    del arg196_1
    del buf300
    buf302 = empty((8, 768), device='cpu', dtype=torch.float32)
    buf303 = empty_strided((8, 1), (1, 8), device='cpu', dtype=torch.float32)
    buf304 = empty_strided((8, 1), (1, 8), device='cpu', dtype=torch.float32)
    buf306 = empty((8, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mean_native_layer_norm_73(c_void_p(buf277.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()))
    del arg111_1
    del arg118_1
    del arg121_1
    del arg122_1
    del buf277
    del buf294
    del buf301
    del buf302
    del buf303
    del buf304
    buf307 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_188, x_190, x_192], Original ATen: [aten.addmm, aten.mean, aten.native_layer_norm]
    extern_kernels.addmm(arg198_1, buf306, reinterpret_tensor(arg197_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf307)
    del arg197_1
    del arg198_1
    return (buf307, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((732, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((732, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((732, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((732, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((732, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((732, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((732, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((732, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((732, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((732, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((732, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((732, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((197, 197), (197, 1), device='cpu', dtype=torch.int64)
    arg201_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((197, 197), (197, 1), device='cpu', dtype=torch.int64)
    arg203_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((197, 197), (197, 1), device='cpu', dtype=torch.int64)
    arg205_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((197, 197), (197, 1), device='cpu', dtype=torch.int64)
    arg207_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((197, 197), (197, 1), device='cpu', dtype=torch.int64)
    arg209_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((197, 197), (197, 1), device='cpu', dtype=torch.int64)
    arg211_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((197, 197), (197, 1), device='cpu', dtype=torch.int64)
    arg213_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((197, 197), (197, 1), device='cpu', dtype=torch.int64)
    arg215_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((197, 197), (197, 1), device='cpu', dtype=torch.int64)
    arg217_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((197, 197), (197, 1), device='cpu', dtype=torch.int64)
    arg219_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((197, 197), (197, 1), device='cpu', dtype=torch.int64)
    arg221_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((197, 197), (197, 1), device='cpu', dtype=torch.int64)
    arg223_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('beit_base_patch16_224', benchmark_compiled_module)
