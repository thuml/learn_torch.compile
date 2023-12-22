
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(57600L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (57600L*x1) + (172800L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (172800L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (144L*x1) + (432L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (432L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1)));
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
                            auto tmp9 = static_cast<int>(401);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp17 = tmp15 + tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                                return tmp22;
                            }
                            ;
                            auto tmp23 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp24 = decltype(tmp20)::blendv(tmp23, tmp20, tmp14);
                            auto tmp25 = tmp24 + tmp16;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp25);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1)));
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (401L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (401L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
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
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(128.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_index_add_cat_convolution_mul_native_layer_norm_sub_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr17,
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20,
                       float* out_ptr21,
                       float* out_ptr23,
                       float* out_ptr24,
                       float* out_ptr25,
                       float* out_ptr26,
                       float* out_ptr27)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
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
                            auto tmp9 = static_cast<int>(401);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp17 = tmp15 + tmp16;
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp22)::blendv(tmp25, tmp22, tmp14);
                            auto tmp27 = tmp26 + tmp16;
                            auto tmp28 = tmp27 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp28);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = static_cast<float>(-0.75);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(-3.75);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = static_cast<float>(-6.0);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
                    auto tmp19 = static_cast<float>(-3.0);
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    out_ptr2[static_cast<long>(x0)] = tmp20;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.25);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = static_cast<float>(2.25);
                    auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                    auto tmp13 = decltype(tmp12)(tmp12 * tmp8);
                    auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
                    auto tmp15 = static_cast<float>(1.0);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    out_ptr3[static_cast<long>(x0)] = tmp16;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
                    auto tmp11 = static_cast<float>(1.25);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(2.25);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp10);
                    auto tmp17 = decltype(tmp16)(tmp16 + tmp9);
                    out_ptr4[static_cast<long>(x0)] = tmp17;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
                    auto tmp11 = static_cast<float>(-0.75);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(-3.75);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = static_cast<float>(-6.0);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
                    auto tmp19 = static_cast<float>(-3.0);
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    out_ptr5[static_cast<long>(x0)] = tmp20;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = static_cast<float>(-0.75);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(-3.75);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = static_cast<float>(-6.0);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
                    auto tmp19 = static_cast<float>(-3.0);
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    out_ptr6[static_cast<long>(x0)] = tmp20;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.25);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = static_cast<float>(2.25);
                    auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                    auto tmp13 = decltype(tmp12)(tmp12 * tmp8);
                    auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
                    auto tmp15 = static_cast<float>(1.0);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    out_ptr7[static_cast<long>(x0)] = tmp16;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
                    auto tmp11 = static_cast<float>(1.25);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(2.25);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp10);
                    auto tmp17 = decltype(tmp16)(tmp16 + tmp9);
                    out_ptr8[static_cast<long>(x0)] = tmp17;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
                    auto tmp11 = static_cast<float>(-0.75);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(-3.75);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = static_cast<float>(-6.0);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
                    auto tmp19 = static_cast<float>(-3.0);
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    out_ptr9[static_cast<long>(x0)] = tmp20;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = static_cast<float>(-0.75);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(-3.75);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = static_cast<float>(-6.0);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
                    auto tmp19 = static_cast<float>(-3.0);
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    out_ptr10[static_cast<long>(x0)] = tmp20;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.25);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = static_cast<float>(2.25);
                    auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                    auto tmp13 = decltype(tmp12)(tmp12 * tmp8);
                    auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
                    auto tmp15 = static_cast<float>(1.0);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    out_ptr11[static_cast<long>(x0)] = tmp16;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
                    auto tmp11 = static_cast<float>(1.25);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(2.25);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp10);
                    auto tmp17 = decltype(tmp16)(tmp16 + tmp9);
                    out_ptr12[static_cast<long>(x0)] = tmp17;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
                    auto tmp11 = static_cast<float>(-0.75);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(-3.75);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = static_cast<float>(-6.0);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
                    auto tmp19 = static_cast<float>(-3.0);
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    out_ptr13[static_cast<long>(x0)] = tmp20;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(224L); x2+=static_cast<long>(1L))
                    {
                        auto tmp26 = out_ptr2[static_cast<long>(x2)];
                        auto tmp31 = out_ptr3[static_cast<long>(x2)];
                        auto tmp38 = out_ptr6[static_cast<long>(x2)];
                        auto tmp41 = out_ptr7[static_cast<long>(x2)];
                        auto tmp49 = out_ptr10[static_cast<long>(x2)];
                        auto tmp52 = out_ptr11[static_cast<long>(x2)];
                        auto tmp59 = out_ptr4[static_cast<long>(x2)];
                        auto tmp66 = out_ptr5[static_cast<long>(x2)];
                        auto tmp70 = out_ptr8[static_cast<long>(x2)];
                        auto tmp74 = out_ptr9[static_cast<long>(x2)];
                        auto tmp78 = out_ptr12[static_cast<long>(x2)];
                        auto tmp82 = out_ptr13[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = c10::convert<float>(tmp0);
                        auto tmp2 = static_cast<float>(0.5);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = static_cast<float>(1.0714285714285714);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                        auto tmp7 = std::floor(tmp6);
                        auto tmp8 = c10::convert<long>(tmp7);
                        auto tmp9 = static_cast<long>(1);
                        auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                        auto tmp11 = static_cast<long>(0);
                        auto tmp12 = max_propagate_nan(tmp10, tmp11);
                        auto tmp13 = static_cast<long>(239);
                        auto tmp14 = min_propagate_nan(tmp12, tmp13);
                        auto tmp15 = c10::convert<long>(x2);
                        auto tmp16 = c10::convert<float>(tmp15);
                        auto tmp17 = decltype(tmp16)(tmp16 + tmp2);
                        auto tmp18 = decltype(tmp17)(tmp17 * tmp4);
                        auto tmp19 = decltype(tmp18)(tmp18 - tmp2);
                        auto tmp20 = std::floor(tmp19);
                        auto tmp21 = c10::convert<long>(tmp20);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp9);
                        auto tmp23 = max_propagate_nan(tmp22, tmp11);
                        auto tmp24 = min_propagate_nan(tmp23, tmp13);
                        auto tmp25 = in_ptr4[static_cast<long>(tmp24 + (240L*tmp14) + (57600L*x0))];
                        auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                        auto tmp28 = max_propagate_nan(tmp21, tmp11);
                        auto tmp29 = min_propagate_nan(tmp28, tmp13);
                        auto tmp30 = in_ptr4[static_cast<long>(tmp29 + (240L*tmp14) + (57600L*x0))];
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        auto tmp33 = decltype(tmp27)(tmp27 + tmp32);
                        auto tmp34 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp35 = max_propagate_nan(tmp34, tmp11);
                        auto tmp36 = min_propagate_nan(tmp35, tmp13);
                        auto tmp37 = in_ptr4[static_cast<long>(tmp24 + (240L*tmp36) + (57600L*x0))];
                        auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                        auto tmp40 = in_ptr4[static_cast<long>(tmp29 + (240L*tmp36) + (57600L*x0))];
                        auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                        auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                        auto tmp44 = static_cast<long>(2);
                        auto tmp45 = decltype(tmp8)(tmp8 + tmp44);
                        auto tmp46 = max_propagate_nan(tmp45, tmp11);
                        auto tmp47 = min_propagate_nan(tmp46, tmp13);
                        auto tmp48 = in_ptr4[static_cast<long>(tmp24 + (240L*tmp47) + (57600L*x0))];
                        auto tmp50 = decltype(tmp48)(tmp48 * tmp49);
                        auto tmp51 = in_ptr4[static_cast<long>(tmp29 + (240L*tmp47) + (57600L*x0))];
                        auto tmp53 = decltype(tmp51)(tmp51 * tmp52);
                        auto tmp54 = decltype(tmp50)(tmp50 + tmp53);
                        auto tmp55 = decltype(tmp21)(tmp21 + tmp9);
                        auto tmp56 = max_propagate_nan(tmp55, tmp11);
                        auto tmp57 = min_propagate_nan(tmp56, tmp13);
                        auto tmp58 = in_ptr4[static_cast<long>(tmp57 + (240L*tmp14) + (57600L*x0))];
                        auto tmp60 = decltype(tmp58)(tmp58 * tmp59);
                        auto tmp61 = decltype(tmp33)(tmp33 + tmp60);
                        auto tmp62 = decltype(tmp21)(tmp21 + tmp44);
                        auto tmp63 = max_propagate_nan(tmp62, tmp11);
                        auto tmp64 = min_propagate_nan(tmp63, tmp13);
                        auto tmp65 = in_ptr4[static_cast<long>(tmp64 + (240L*tmp14) + (57600L*x0))];
                        auto tmp67 = decltype(tmp65)(tmp65 * tmp66);
                        auto tmp68 = decltype(tmp61)(tmp61 + tmp67);
                        auto tmp69 = in_ptr4[static_cast<long>(tmp57 + (240L*tmp36) + (57600L*x0))];
                        auto tmp71 = decltype(tmp69)(tmp69 * tmp70);
                        auto tmp72 = decltype(tmp43)(tmp43 + tmp71);
                        auto tmp73 = in_ptr4[static_cast<long>(tmp64 + (240L*tmp36) + (57600L*x0))];
                        auto tmp75 = decltype(tmp73)(tmp73 * tmp74);
                        auto tmp76 = decltype(tmp72)(tmp72 + tmp75);
                        auto tmp77 = in_ptr4[static_cast<long>(tmp57 + (240L*tmp47) + (57600L*x0))];
                        auto tmp79 = decltype(tmp77)(tmp77 * tmp78);
                        auto tmp80 = decltype(tmp54)(tmp54 + tmp79);
                        auto tmp81 = in_ptr4[static_cast<long>(tmp64 + (240L*tmp47) + (57600L*x0))];
                        auto tmp83 = decltype(tmp81)(tmp81 * tmp82);
                        auto tmp84 = decltype(tmp80)(tmp80 + tmp83);
                        in_out_ptr0[static_cast<long>(x2 + (224L*x1) + (50176L*x0))] = tmp68;
                        in_out_ptr1[static_cast<long>(x2 + (224L*x1) + (50176L*x0))] = tmp76;
                        in_out_ptr2[static_cast<long>(x2 + (224L*x1) + (50176L*x0))] = tmp84;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = static_cast<float>(-0.75);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(-3.75);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = static_cast<float>(-6.0);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
                    auto tmp19 = static_cast<float>(-3.0);
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    out_ptr17[static_cast<long>(x0)] = tmp20;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = static_cast<float>(-0.75);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(-3.75);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = static_cast<float>(-6.0);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
                    auto tmp19 = static_cast<float>(-3.0);
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    out_ptr18[static_cast<long>(x0)] = tmp20;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.25);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = static_cast<float>(2.25);
                    auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                    auto tmp13 = decltype(tmp12)(tmp12 * tmp8);
                    auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
                    auto tmp15 = static_cast<float>(1.0);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    out_ptr19[static_cast<long>(x0)] = tmp16;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
                    auto tmp11 = static_cast<float>(1.25);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(2.25);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp10);
                    auto tmp17 = decltype(tmp16)(tmp16 + tmp9);
                    out_ptr20[static_cast<long>(x0)] = tmp17;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
                    auto tmp11 = static_cast<float>(-0.75);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(-3.75);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = static_cast<float>(-6.0);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
                    auto tmp19 = static_cast<float>(-3.0);
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    out_ptr21[static_cast<long>(x0)] = tmp20;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(224L); x2+=static_cast<long>(1L))
                    {
                        auto tmp25 = out_ptr18[static_cast<long>(x2)];
                        auto tmp30 = out_ptr19[static_cast<long>(x2)];
                        auto tmp37 = out_ptr20[static_cast<long>(x2)];
                        auto tmp45 = out_ptr21[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = c10::convert<float>(tmp0);
                        auto tmp2 = static_cast<float>(0.5);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = static_cast<float>(1.0714285714285714);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                        auto tmp7 = std::floor(tmp6);
                        auto tmp8 = c10::convert<long>(tmp7);
                        auto tmp9 = static_cast<long>(0);
                        auto tmp10 = max_propagate_nan(tmp8, tmp9);
                        auto tmp11 = static_cast<long>(239);
                        auto tmp12 = min_propagate_nan(tmp10, tmp11);
                        auto tmp13 = c10::convert<long>(x2);
                        auto tmp14 = c10::convert<float>(tmp13);
                        auto tmp15 = decltype(tmp14)(tmp14 + tmp2);
                        auto tmp16 = decltype(tmp15)(tmp15 * tmp4);
                        auto tmp17 = decltype(tmp16)(tmp16 - tmp2);
                        auto tmp18 = std::floor(tmp17);
                        auto tmp19 = c10::convert<long>(tmp18);
                        auto tmp20 = static_cast<long>(1);
                        auto tmp21 = decltype(tmp19)(tmp19 - tmp20);
                        auto tmp22 = max_propagate_nan(tmp21, tmp9);
                        auto tmp23 = min_propagate_nan(tmp22, tmp11);
                        auto tmp24 = in_ptr4[static_cast<long>(tmp23 + (240L*tmp12) + (57600L*x0))];
                        auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                        auto tmp27 = max_propagate_nan(tmp19, tmp9);
                        auto tmp28 = min_propagate_nan(tmp27, tmp11);
                        auto tmp29 = in_ptr4[static_cast<long>(tmp28 + (240L*tmp12) + (57600L*x0))];
                        auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                        auto tmp32 = decltype(tmp26)(tmp26 + tmp31);
                        auto tmp33 = decltype(tmp19)(tmp19 + tmp20);
                        auto tmp34 = max_propagate_nan(tmp33, tmp9);
                        auto tmp35 = min_propagate_nan(tmp34, tmp11);
                        auto tmp36 = in_ptr4[static_cast<long>(tmp35 + (240L*tmp12) + (57600L*x0))];
                        auto tmp38 = decltype(tmp36)(tmp36 * tmp37);
                        auto tmp39 = decltype(tmp32)(tmp32 + tmp38);
                        auto tmp40 = static_cast<long>(2);
                        auto tmp41 = decltype(tmp19)(tmp19 + tmp40);
                        auto tmp42 = max_propagate_nan(tmp41, tmp9);
                        auto tmp43 = min_propagate_nan(tmp42, tmp11);
                        auto tmp44 = in_ptr4[static_cast<long>(tmp43 + (240L*tmp12) + (57600L*x0))];
                        auto tmp46 = decltype(tmp44)(tmp44 * tmp45);
                        auto tmp47 = decltype(tmp39)(tmp39 + tmp46);
                        in_out_ptr3[static_cast<long>(x2 + (224L*x1) + (50176L*x0))] = tmp47;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.25);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = static_cast<float>(2.25);
                    auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                    auto tmp13 = decltype(tmp12)(tmp12 * tmp8);
                    auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
                    auto tmp15 = static_cast<float>(1.0);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    out_ptr23[static_cast<long>(x0)] = tmp16;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
                    auto tmp11 = static_cast<float>(1.25);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(2.25);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp10);
                    auto tmp17 = decltype(tmp16)(tmp16 + tmp9);
                    out_ptr24[static_cast<long>(x0)] = tmp17;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.0714285714285714);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = std::floor(tmp6);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
                    auto tmp11 = static_cast<float>(-0.75);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(-3.75);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                    auto tmp16 = static_cast<float>(-6.0);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
                    auto tmp19 = static_cast<float>(-3.0);
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    out_ptr25[static_cast<long>(x0)] = tmp20;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(224L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(224L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (224L*x2) + (50176L*x1) + (150528L*x0)));
                            auto tmp1 = out_ptr17[static_cast<long>(x2)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x3 + (224L*x2) + (50176L*x1) + (150528L*x0)));
                            auto tmp5 = out_ptr23[static_cast<long>(x2)];
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x3 + (224L*x2) + (50176L*x1) + (150528L*x0)));
                            auto tmp10 = out_ptr24[static_cast<long>(x2)];
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x3 + (224L*x2) + (50176L*x1) + (150528L*x0)));
                            auto tmp15 = out_ptr25[static_cast<long>(x2)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = tmp3 + tmp7;
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp8 + tmp12;
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp14 * tmp16;
                            auto tmp18 = tmp13 + tmp17;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr26[static_cast<long>(x1 + (3L*x3) + (3L*x3_inner) + (672L*x2) + (150528L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr5[static_cast<long>(x2 + (256L*x1) + (768L*x0))];
                        out_ptr27[static_cast<long>(x1 + (3L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1)));
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
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp17 = tmp15 + tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                                return tmp22;
                            }
                            ;
                            auto tmp23 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp24 = decltype(tmp20)::blendv(tmp23, tmp20, tmp14);
                            auto tmp25 = tmp24 + tmp16;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp25);
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1)));
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
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
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(256.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
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
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp17 = tmp15 + tmp16;
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp22)::blendv(tmp25, tmp22, tmp14);
                            auto tmp27 = tmp26 + tmp16;
                            auto tmp28 = tmp27 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp28);
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
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
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(256.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
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
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-06);
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


cpp_fused_gelu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_9 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp20 = in_ptr8[static_cast<long>(x1 + (401L*x0))];
                        auto tmp23 = in_ptr9[static_cast<long>(x1 + (401L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr4 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr5 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(128.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1231872L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_12 = async_compile.cpp('''
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
                       const float* in_ptr10,
                       const float* in_ptr11,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
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
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
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
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = in_ptr8[static_cast<long>(x0)];
                    auto tmp10 = in_ptr9[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(256.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_gelu_native_layer_norm_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(403456L); x0+=static_cast<long>(8L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (51328L*x0)));
                        auto tmp1 = in_ptr5[static_cast<long>(x0)];
                        auto tmp4 = in_ptr6[static_cast<long>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(128.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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
                    tmp11.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp4));
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
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
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 - tmp17;
                        auto tmp20 = static_cast<float>(256.0);
                        auto tmp21 = tmp19 / tmp20;
                        auto tmp22 = static_cast<float>(1e-06);
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        auto tmp24 = 1 / std::sqrt(tmp23);
                        auto tmp25 = at::vec::Vectorized<float>(tmp24);
                        auto tmp26 = tmp18 * tmp25;
                        auto tmp28 = tmp26 * tmp27;
                        auto tmp30 = tmp28 + tmp29;
                        tmp30.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (50432L*x0)), static_cast<long>(256L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (50432L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (50432L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (50432L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused_clone_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (256L*x2) + (50432L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (50432L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_native_layer_norm_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5)
{
    auto out_ptr4 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<float> tmp_acc1 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                Welford<float> tmp_acc2 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc2_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp1 >= tmp1;
                    auto tmp3 = static_cast<int>(1);
                    auto tmp4 = tmp1 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = masked_load(in_ptr1 + static_cast<long>(x1 + (256L*x0)), to_float_mask(tmp4));
                        return tmp6;
                    }
                    ;
                    auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                    auto tmp8 = tmp1 >= tmp3;
                    auto tmp9 = static_cast<int>(197);
                    auto tmp10 = tmp1 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x1 + (50432L*x0)), to_float_mask(tmp8));
                        return tmp12;
                    }
                    ;
                    auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                    auto tmp14 = to_float_mask(tmp4);
                    auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = masked_load(in_ptr1 + static_cast<long>(x1 + (256L*x0)), to_float_mask(tmp4));
                        return tmp19;
                    }
                    ;
                    auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                    auto tmp21 = [&]
                    {
                        auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x1 + (50432L*x0)), to_float_mask(tmp8));
                        return tmp22;
                    }
                    ;
                    auto tmp23 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                    auto tmp24 = decltype(tmp20)::blendv(tmp23, tmp20, tmp14);
                    auto tmp25 = tmp24 + tmp16;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp17);
                    tmp_acc2_vec = welford_combine(tmp_acc2_vec, tmp25);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.mean);
                tmp_acc2 = welford_combine(tmp_acc2, welford_vec_reduce_all(tmp_acc2_vec));
                out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc2.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp4 = out_ptr1[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                auto tmp33 = out_ptr2[static_cast<long>(x0)];
                auto tmp36 = out_ptr3[static_cast<long>(x0)];
                auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp44 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(256.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = static_cast<int>(0);
                auto tmp17 = tmp16 >= tmp16;
                auto tmp18 = static_cast<int>(1);
                auto tmp19 = tmp16 < tmp18;
                auto tmp20 = [&]
                {
                    auto tmp21 = masked_load(in_ptr1 + static_cast<long>(x1 + (256L*x0)), to_float_mask(tmp19));
                    return tmp21;
                }
                ;
                auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp19));
                auto tmp23 = tmp16 >= tmp18;
                auto tmp24 = static_cast<int>(197);
                auto tmp25 = tmp16 < tmp24;
                auto tmp26 = [&]
                {
                    auto tmp27 = masked_load(in_ptr0 + static_cast<long>(x1 + (50432L*x0)), to_float_mask(tmp23));
                    return tmp27;
                }
                ;
                auto tmp28 = decltype(tmp26())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp26(), to_float_mask(tmp23));
                auto tmp29 = to_float_mask(tmp19);
                auto tmp30 = decltype(tmp22)::blendv(tmp28, tmp22, tmp29);
                auto tmp32 = tmp30 + tmp31;
                auto tmp34 = at::vec::Vectorized<float>(tmp33);
                auto tmp35 = tmp32 - tmp34;
                auto tmp37 = tmp36 / tmp5;
                auto tmp38 = decltype(tmp37)(tmp37 + tmp7);
                auto tmp39 = 1 / std::sqrt(tmp38);
                auto tmp40 = at::vec::Vectorized<float>(tmp39);
                auto tmp41 = tmp35 * tmp40;
                auto tmp43 = tmp41 * tmp42;
                auto tmp45 = tmp43 + tmp44;
                tmp15.store(out_ptr4 + static_cast<long>(x1 + (256L*x0)));
                tmp45.store(out_ptr5 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
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
''')


cpp_fused_gelu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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
''')


cpp_fused_cat_native_layer_norm_22 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc2 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc2_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc3 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc3_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(401);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp4));
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp8));
                            auto tmp22 = decltype(tmp18)::blendv(tmp21, tmp18, tmp14);
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp4));
                            auto tmp26 = [&]
                            {
                                auto tmp27 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                return tmp27;
                            }
                            ;
                            auto tmp28 = decltype(tmp26())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp26(), to_float_mask(tmp8));
                            auto tmp29 = decltype(tmp25)::blendv(tmp28, tmp25, tmp14);
                            auto tmp30 = [&]
                            {
                                auto tmp31 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                                return tmp31;
                            }
                            ;
                            auto tmp32 = decltype(tmp30())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp30(), to_float_mask(tmp4));
                            auto tmp33 = [&]
                            {
                                auto tmp34 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                return tmp34;
                            }
                            ;
                            auto tmp35 = decltype(tmp33())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp33(), to_float_mask(tmp8));
                            auto tmp36 = decltype(tmp32)::blendv(tmp35, tmp32, tmp14);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp15);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp22);
                            tmp_acc2_vec = welford_combine(tmp_acc2_vec, tmp29);
                            tmp_acc3_vec = welford_combine(tmp_acc3_vec, tmp36);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc1.m2);
                        tmp_acc2 = welford_combine(tmp_acc2, welford_vec_reduce_all(tmp_acc2_vec));
                        out_ptr2[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc2.mean);
                        tmp_acc3 = welford_combine(tmp_acc3, welford_vec_reduce_all(tmp_acc3_vec));
                        out_ptr3[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc3.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = out_ptr0[static_cast<long>(x1 + (401L*x0))];
                        auto tmp19 = out_ptr1[static_cast<long>(x1 + (401L*x0))];
                        auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp38 = out_ptr2[static_cast<long>(x1 + (401L*x0))];
                        auto tmp41 = out_ptr3[static_cast<long>(x1 + (401L*x0))];
                        auto tmp47 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp49 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 - tmp17;
                        auto tmp20 = static_cast<float>(128.0);
                        auto tmp21 = tmp19 / tmp20;
                        auto tmp22 = static_cast<float>(1e-06);
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        auto tmp24 = 1 / std::sqrt(tmp23);
                        auto tmp25 = at::vec::Vectorized<float>(tmp24);
                        auto tmp26 = tmp18 * tmp25;
                        auto tmp28 = tmp26 * tmp27;
                        auto tmp30 = tmp28 + tmp29;
                        auto tmp31 = [&]
                        {
                            auto tmp32 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                            return tmp32;
                        }
                        ;
                        auto tmp33 = decltype(tmp31())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp31(), to_float_mask(tmp4));
                        auto tmp34 = [&]
                        {
                            auto tmp35 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            return tmp35;
                        }
                        ;
                        auto tmp36 = decltype(tmp34())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp34(), to_float_mask(tmp8));
                        auto tmp37 = decltype(tmp33)::blendv(tmp36, tmp33, tmp14);
                        auto tmp39 = at::vec::Vectorized<float>(tmp38);
                        auto tmp40 = tmp37 - tmp39;
                        auto tmp42 = tmp41 / tmp20;
                        auto tmp43 = decltype(tmp42)(tmp42 + tmp22);
                        auto tmp44 = 1 / std::sqrt(tmp43);
                        auto tmp45 = at::vec::Vectorized<float>(tmp44);
                        auto tmp46 = tmp40 * tmp45;
                        auto tmp48 = tmp46 * tmp47;
                        auto tmp50 = tmp48 + tmp49;
                        tmp30.store(out_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        tmp50.store(out_ptr5 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(400L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (51328L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (401L*x1) + (401L*x1_inner) + (51328L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(400L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (51328L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (401L*x1) + (401L*x1_inner) + (51328L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (401L*x0))];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (401L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (401L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (401L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (401L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x1) + (128L*x2) + (51328L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (12832L*x1) + (51328L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(401);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp17 = tmp15 + tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                return tmp22;
                            }
                            ;
                            auto tmp23 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp24 = decltype(tmp20)::blendv(tmp23, tmp20, tmp14);
                            auto tmp25 = tmp24 + tmp16;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp25);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (401L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (401L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(128.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1231872L); x0+=static_cast<long>(8L))
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


cpp_fused_add_gelu_native_layer_norm_28 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr5)
{
    auto out_ptr4 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<float> tmp_acc1 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                Welford<float> tmp_acc2 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc2_vec = Welford<at::vec::Vectorized<float>>();
                Welford<float> tmp_acc3 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc3_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp0 = static_cast<int>(0);
                    auto tmp1 = tmp0 >= tmp0;
                    auto tmp2 = static_cast<int>(1);
                    auto tmp3 = tmp0 < tmp2;
                    auto tmp4 = [&]
                    {
                        auto tmp5 = masked_load(in_ptr0 + static_cast<long>(x1 + (128L*x0)), to_float_mask(tmp3));
                        return tmp5;
                    }
                    ;
                    auto tmp6 = decltype(tmp4())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp4(), to_float_mask(tmp3));
                    auto tmp7 = tmp0 >= tmp2;
                    auto tmp8 = static_cast<int>(401);
                    auto tmp9 = tmp0 < tmp8;
                    auto tmp10 = [&]
                    {
                        auto tmp11 = masked_load(in_ptr1 + static_cast<long>(x1 + (51328L*x0)), to_float_mask(tmp7));
                        return tmp11;
                    }
                    ;
                    auto tmp12 = decltype(tmp10())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp10(), to_float_mask(tmp7));
                    auto tmp13 = to_float_mask(tmp3);
                    auto tmp14 = decltype(tmp6)::blendv(tmp12, tmp6, tmp13);
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = [&]
                    {
                        auto tmp18 = masked_load(in_ptr0 + static_cast<long>(x1 + (128L*x0)), to_float_mask(tmp3));
                        return tmp18;
                    }
                    ;
                    auto tmp19 = decltype(tmp17())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp17(), to_float_mask(tmp3));
                    auto tmp20 = [&]
                    {
                        auto tmp21 = masked_load(in_ptr1 + static_cast<long>(x1 + (51328L*x0)), to_float_mask(tmp7));
                        return tmp21;
                    }
                    ;
                    auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp7));
                    auto tmp23 = decltype(tmp19)::blendv(tmp22, tmp19, tmp13);
                    auto tmp24 = tmp23 + tmp15;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = masked_load(in_ptr3 + static_cast<long>(x1 + (128L*x0)), to_float_mask(tmp3));
                        return tmp26;
                    }
                    ;
                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp3));
                    auto tmp28 = [&]
                    {
                        auto tmp29 = masked_load(in_ptr1 + static_cast<long>(x1 + (51328L*x0)), to_float_mask(tmp7));
                        return tmp29;
                    }
                    ;
                    auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp7));
                    auto tmp31 = decltype(tmp27)::blendv(tmp30, tmp27, tmp13);
                    auto tmp33 = tmp31 + tmp32;
                    auto tmp35 = tmp33 + tmp34;
                    auto tmp36 = [&]
                    {
                        auto tmp37 = masked_load(in_ptr3 + static_cast<long>(x1 + (128L*x0)), to_float_mask(tmp3));
                        return tmp37;
                    }
                    ;
                    auto tmp38 = decltype(tmp36())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp36(), to_float_mask(tmp3));
                    auto tmp39 = [&]
                    {
                        auto tmp40 = masked_load(in_ptr1 + static_cast<long>(x1 + (51328L*x0)), to_float_mask(tmp7));
                        return tmp40;
                    }
                    ;
                    auto tmp41 = decltype(tmp39())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp39(), to_float_mask(tmp7));
                    auto tmp42 = decltype(tmp38)::blendv(tmp41, tmp38, tmp13);
                    auto tmp43 = tmp42 + tmp32;
                    auto tmp44 = tmp43 + tmp34;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp16);
                    tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp24);
                    tmp_acc2_vec = welford_combine(tmp_acc2_vec, tmp35);
                    tmp_acc3_vec = welford_combine(tmp_acc3_vec, tmp44);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                tmp_acc2 = welford_combine(tmp_acc2, welford_vec_reduce_all(tmp_acc2_vec));
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc2.mean);
                tmp_acc3 = welford_combine(tmp_acc3, welford_vec_reduce_all(tmp_acc3_vec));
                out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc3.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                auto tmp17 = out_ptr0[static_cast<long>(x0)];
                auto tmp20 = out_ptr1[static_cast<long>(x0)];
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp39 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp41 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp43 = out_ptr2[static_cast<long>(x0)];
                auto tmp46 = out_ptr3[static_cast<long>(x0)];
                auto tmp52 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                auto tmp54 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                auto tmp0 = static_cast<int>(0);
                auto tmp1 = tmp0 >= tmp0;
                auto tmp2 = static_cast<int>(1);
                auto tmp3 = tmp0 < tmp2;
                auto tmp4 = [&]
                {
                    auto tmp5 = masked_load(in_ptr0 + static_cast<long>(x1 + (128L*x0)), to_float_mask(tmp3));
                    return tmp5;
                }
                ;
                auto tmp6 = decltype(tmp4())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp4(), to_float_mask(tmp3));
                auto tmp7 = tmp0 >= tmp2;
                auto tmp8 = static_cast<int>(401);
                auto tmp9 = tmp0 < tmp8;
                auto tmp10 = [&]
                {
                    auto tmp11 = masked_load(in_ptr1 + static_cast<long>(x1 + (51328L*x0)), to_float_mask(tmp7));
                    return tmp11;
                }
                ;
                auto tmp12 = decltype(tmp10())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp10(), to_float_mask(tmp7));
                auto tmp13 = to_float_mask(tmp3);
                auto tmp14 = decltype(tmp6)::blendv(tmp12, tmp6, tmp13);
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = static_cast<float>(128.0);
                auto tmp22 = tmp20 / tmp21;
                auto tmp23 = static_cast<float>(1e-06);
                auto tmp24 = decltype(tmp22)(tmp22 + tmp23);
                auto tmp25 = 1 / std::sqrt(tmp24);
                auto tmp26 = at::vec::Vectorized<float>(tmp25);
                auto tmp27 = tmp19 * tmp26;
                auto tmp29 = tmp27 * tmp28;
                auto tmp31 = tmp29 + tmp30;
                auto tmp32 = [&]
                {
                    auto tmp33 = masked_load(in_ptr3 + static_cast<long>(x1 + (128L*x0)), to_float_mask(tmp3));
                    return tmp33;
                }
                ;
                auto tmp34 = decltype(tmp32())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp32(), to_float_mask(tmp3));
                auto tmp35 = [&]
                {
                    auto tmp36 = masked_load(in_ptr1 + static_cast<long>(x1 + (51328L*x0)), to_float_mask(tmp7));
                    return tmp36;
                }
                ;
                auto tmp37 = decltype(tmp35())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp35(), to_float_mask(tmp7));
                auto tmp38 = decltype(tmp34)::blendv(tmp37, tmp34, tmp13);
                auto tmp40 = tmp38 + tmp39;
                auto tmp42 = tmp40 + tmp41;
                auto tmp44 = at::vec::Vectorized<float>(tmp43);
                auto tmp45 = tmp42 - tmp44;
                auto tmp47 = tmp46 / tmp21;
                auto tmp48 = decltype(tmp47)(tmp47 + tmp23);
                auto tmp49 = 1 / std::sqrt(tmp48);
                auto tmp50 = at::vec::Vectorized<float>(tmp49);
                auto tmp51 = tmp45 * tmp50;
                auto tmp53 = tmp51 * tmp52;
                auto tmp55 = tmp53 + tmp54;
                tmp31.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                tmp55.store(out_ptr5 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
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
''')


cpp_fused_cat_native_layer_norm_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp4));
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
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
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 - tmp17;
                        auto tmp20 = static_cast<float>(256.0);
                        auto tmp21 = tmp19 / tmp20;
                        auto tmp22 = static_cast<float>(1e-06);
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        auto tmp24 = 1 / std::sqrt(tmp23);
                        auto tmp25 = at::vec::Vectorized<float>(tmp24);
                        auto tmp26 = tmp18 * tmp25;
                        auto tmp28 = tmp26 * tmp27;
                        auto tmp30 = tmp28 + tmp29;
                        tmp30.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp17 = tmp15 + tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                return tmp22;
                            }
                            ;
                            auto tmp23 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp24 = decltype(tmp20)::blendv(tmp23, tmp20, tmp14);
                            auto tmp25 = tmp24 + tmp16;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp25);
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(256.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp17 = tmp15 + tmp16;
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp22)::blendv(tmp25, tmp22, tmp14);
                            auto tmp27 = tmp26 + tmp16;
                            auto tmp28 = tmp27 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp28);
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
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
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(256.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-06);
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


cpp_fused_add_native_layer_norm_36 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_gelu_native_layer_norm_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (50432L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
''')


cpp_fused_gelu_native_layer_norm_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp7 = in_ptr4[static_cast<long>(x0)];
                auto tmp10 = in_ptr5[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp11 = static_cast<float>(256.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-06);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                auto tmp21 = tmp19 + tmp20;
                tmp21.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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
''')


cpp_fused_cat_native_layer_norm_40 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = c10::convert<int>(x1);
                            auto tmp13 = static_cast<int>(0);
                            auto tmp14 = tmp12 >= tmp13;
                            auto tmp15 = static_cast<int>(1);
                            auto tmp16 = tmp12 < tmp15;
                            auto tmp18 = tmp16 & tmp8;
                            auto tmp17 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp18));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp17())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp17(), to_float_mask(tmp18));
                            auto tmp21 = tmp12 >= tmp15;
                            auto tmp22 = static_cast<int>(401);
                            auto tmp23 = tmp12 < tmp22;
                            auto tmp25 = tmp21 & tmp8;
                            auto tmp24 = [&]
                            {
                                auto tmp26 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp25));
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp24())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp24(), to_float_mask(tmp25));
                            auto tmp28 = to_float_mask(tmp16);
                            auto tmp29 = decltype(tmp20)::blendv(tmp27, tmp20, tmp28);
                            auto tmp30 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp31 = tmp29 + tmp30;
                            auto tmp32 = masked_load(in_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp33 = tmp31 + tmp32;
                            return tmp33;
                        }
                        ;
                        auto tmp34 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp35 = to_float_mask(tmp4);
                        auto tmp36 = decltype(tmp7)::blendv(tmp34, tmp7, tmp35);
                        tmp36.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(128.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
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
}
''')


cpp_fused_add_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(400L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (51328L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (401L*x1) + (401L*x1_inner) + (51328L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(400L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (51328L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (401L*x1) + (401L*x1_inner) + (51328L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (401L*x0))];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (401L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (401L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (401L*x0))];
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (401L*x0))] = tmp2;
            }
        }
    }
}
''')


cpp_fused_clone_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (128L*x2) + (51328L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (12832L*x1) + (51328L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_native_layer_norm_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp6 = out_ptr1[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 - tmp4;
                auto tmp7 = static_cast<float>(128.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-06);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
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
''')


cpp_fused_cat_native_layer_norm_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            auto tmp17 = masked_load(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp18 = tmp16 + tmp17;
                            return tmp18;
                        }
                        ;
                        auto tmp19 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp20 = to_float_mask(tmp4);
                        auto tmp21 = decltype(tmp7)::blendv(tmp19, tmp7, tmp20);
                        auto tmp22 = [&]
                        {
                            auto tmp23 = masked_load(in_ptr5 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp23;
                        }
                        ;
                        auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp4));
                        auto tmp25 = [&]
                        {
                            auto tmp26 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp27 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp28 = tmp26 + tmp27;
                            auto tmp29 = masked_load(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp30 = tmp28 + tmp29;
                            auto tmp31 = masked_load(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp32 = tmp30 + tmp31;
                            return tmp32;
                        }
                        ;
                        auto tmp33 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp8));
                        auto tmp34 = decltype(tmp24)::blendv(tmp33, tmp24, tmp20);
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        tmp34.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = out_ptr3[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (50432L*x0)), static_cast<long>(256L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (50432L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (50432L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (50432L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (256L*x2) + (50432L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (50432L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_native_layer_norm_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
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
                auto tmp9 = static_cast<float>(1e-06);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
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
''')


cpp_fused_cat_native_layer_norm_51 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = c10::convert<int>(x1);
                            auto tmp13 = static_cast<int>(0);
                            auto tmp14 = tmp12 >= tmp13;
                            auto tmp15 = static_cast<int>(1);
                            auto tmp16 = tmp12 < tmp15;
                            auto tmp18 = tmp16 & tmp8;
                            auto tmp17 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp18));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp17())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp17(), to_float_mask(tmp18));
                            auto tmp21 = tmp12 >= tmp15;
                            auto tmp22 = static_cast<int>(401);
                            auto tmp23 = tmp12 < tmp22;
                            auto tmp25 = tmp21 & tmp8;
                            auto tmp24 = [&]
                            {
                                auto tmp26 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp25));
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp24())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp24(), to_float_mask(tmp25));
                            auto tmp28 = to_float_mask(tmp16);
                            auto tmp29 = decltype(tmp20)::blendv(tmp27, tmp20, tmp28);
                            auto tmp30 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp31 = tmp29 + tmp30;
                            auto tmp32 = masked_load(in_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp33 = tmp31 + tmp32;
                            return tmp33;
                        }
                        ;
                        auto tmp34 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp35 = to_float_mask(tmp4);
                        auto tmp36 = decltype(tmp7)::blendv(tmp34, tmp7, tmp35);
                        tmp36.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(128.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
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
}
''')


cpp_fused_add_native_layer_norm_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = out_ptr3[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-06);
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


cpp_fused_gelu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_55 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
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
                       const float* in_ptr5,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(256.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(403456L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = in_ptr5[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(128.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1231872L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       float* out_ptr2)
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (51328L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = in_ptr5[static_cast<long>(x0)];
                    auto tmp6 = in_ptr6[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
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


cpp_fused_gelu_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_gelu_native_layer_norm_63 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (50432L*x0)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp5 = in_ptr6[static_cast<long>(x0)];
                auto tmp8 = in_ptr7[static_cast<long>(x0)];
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 - tmp6;
                auto tmp9 = static_cast<float>(128.0);
                auto tmp10 = tmp8 / tmp9;
                auto tmp11 = static_cast<float>(1e-06);
                auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                auto tmp13 = 1 / std::sqrt(tmp12);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp7 * tmp14;
                auto tmp17 = tmp15 * tmp16;
                auto tmp19 = tmp17 + tmp18;
                tmp19.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
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
''')


cpp_fused_cat_native_layer_norm_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                auto tmp16 = tmp14 + tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp18 = to_float_mask(tmp4);
                            auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                auto tmp27 = masked_load(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                auto tmp28 = tmp26 + tmp27;
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp30 = decltype(tmp22)::blendv(tmp29, tmp22, tmp18);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp30);
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
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
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(256.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (50432L*x0)), static_cast<long>(256L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (50432L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (50432L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (50432L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (256L*x2) + (50432L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (50432L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_native_layer_norm_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto out_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp26 = in_ptr4[static_cast<long>(x0)];
                    auto tmp29 = in_ptr5[static_cast<long>(x0)];
                    auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp39 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp0 = static_cast<int>(0);
                    auto tmp1 = tmp0 >= tmp0;
                    auto tmp2 = static_cast<int>(1);
                    auto tmp3 = tmp0 < tmp2;
                    auto tmp4 = [&]
                    {
                        auto tmp5 = masked_load(in_ptr0 + static_cast<long>(x1 + (256L*x0)), to_float_mask(tmp3));
                        return tmp5;
                    }
                    ;
                    auto tmp6 = decltype(tmp4())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp4(), to_float_mask(tmp3));
                    auto tmp7 = tmp0 >= tmp2;
                    auto tmp8 = static_cast<int>(197);
                    auto tmp9 = tmp0 < tmp8;
                    auto tmp10 = [&]
                    {
                        auto tmp11 = masked_load(in_ptr1 + static_cast<long>(x1 + (50432L*x0)), to_float_mask(tmp7));
                        auto tmp12 = masked_load(in_ptr2 + static_cast<long>(x1 + (50432L*x0)), to_float_mask(tmp7));
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = masked_load(in_ptr3 + static_cast<long>(x1 + (50432L*x0)), to_float_mask(tmp7));
                        auto tmp15 = tmp13 + tmp14;
                        return tmp15;
                    }
                    ;
                    auto tmp16 = decltype(tmp10())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp10(), to_float_mask(tmp7));
                    auto tmp17 = to_float_mask(tmp3);
                    auto tmp18 = decltype(tmp6)::blendv(tmp16, tmp6, tmp17);
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp23 = tmp21 + tmp22;
                    auto tmp25 = tmp23 + tmp24;
                    auto tmp27 = at::vec::Vectorized<float>(tmp26);
                    auto tmp28 = tmp25 - tmp27;
                    auto tmp30 = static_cast<float>(256.0);
                    auto tmp31 = tmp29 / tmp30;
                    auto tmp32 = static_cast<float>(1e-06);
                    auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                    auto tmp34 = 1 / std::sqrt(tmp33);
                    auto tmp35 = at::vec::Vectorized<float>(tmp34);
                    auto tmp36 = tmp28 * tmp35;
                    auto tmp38 = tmp36 * tmp37;
                    auto tmp40 = tmp38 + tmp39;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp40.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp20);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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
            tmp11.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_gelu_native_layer_norm_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp4 = in_ptr2[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(256.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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
''')


cpp_fused_cat_native_layer_norm_70 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc2 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc2_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc3 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc3_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(401);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                auto tmp16 = tmp14 + tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp18 = to_float_mask(tmp4);
                            auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                auto tmp27 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                auto tmp28 = tmp26 + tmp27;
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp30 = decltype(tmp22)::blendv(tmp29, tmp22, tmp18);
                            auto tmp31 = [&]
                            {
                                auto tmp32 = masked_load(in_ptr4 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                                return tmp32;
                            }
                            ;
                            auto tmp33 = decltype(tmp31())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp31(), to_float_mask(tmp4));
                            auto tmp34 = [&]
                            {
                                auto tmp35 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                auto tmp36 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                auto tmp37 = tmp35 + tmp36;
                                auto tmp38 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                auto tmp39 = tmp37 + tmp38;
                                return tmp39;
                            }
                            ;
                            auto tmp40 = decltype(tmp34())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp34(), to_float_mask(tmp8));
                            auto tmp41 = decltype(tmp33)::blendv(tmp40, tmp33, tmp18);
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(in_ptr4 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp4));
                            auto tmp45 = [&]
                            {
                                auto tmp46 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                auto tmp47 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                auto tmp48 = tmp46 + tmp47;
                                auto tmp49 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                                auto tmp50 = tmp48 + tmp49;
                                return tmp50;
                            }
                            ;
                            auto tmp51 = decltype(tmp45())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp45(), to_float_mask(tmp8));
                            auto tmp52 = decltype(tmp44)::blendv(tmp51, tmp44, tmp18);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp30);
                            tmp_acc2_vec = welford_combine(tmp_acc2_vec, tmp41);
                            tmp_acc3_vec = welford_combine(tmp_acc3_vec, tmp52);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc1.m2);
                        tmp_acc2 = welford_combine(tmp_acc2, welford_vec_reduce_all(tmp_acc2_vec));
                        out_ptr2[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc2.mean);
                        tmp_acc3 = welford_combine(tmp_acc3, welford_vec_reduce_all(tmp_acc3_vec));
                        out_ptr3[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc3.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp20 = out_ptr0[static_cast<long>(x1 + (401L*x0))];
                        auto tmp23 = out_ptr1[static_cast<long>(x1 + (401L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(128.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(400L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (51328L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (401L*x1) + (401L*x1_inner) + (51328L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(400L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (51328L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (401L*x1) + (401L*x1_inner) + (51328L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (401L*x0))];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (401L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (401L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (401L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (401L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x1) + (128L*x2) + (51328L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (12832L*x1) + (51328L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_gelu_native_layer_norm_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr10,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto out_ptr3 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp32 = in_ptr5[static_cast<long>(401L*x0)];
                    auto tmp35 = in_ptr6[static_cast<long>(401L*x0)];
                    auto tmp43 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp45 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp0 = static_cast<int>(0);
                    auto tmp1 = tmp0 >= tmp0;
                    auto tmp2 = static_cast<int>(1);
                    auto tmp3 = tmp0 < tmp2;
                    auto tmp4 = [&]
                    {
                        auto tmp5 = masked_load(in_ptr0 + static_cast<long>(x1 + (128L*x0)), to_float_mask(tmp3));
                        return tmp5;
                    }
                    ;
                    auto tmp6 = decltype(tmp4())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp4(), to_float_mask(tmp3));
                    auto tmp7 = tmp0 >= tmp2;
                    auto tmp8 = static_cast<int>(401);
                    auto tmp9 = tmp0 < tmp8;
                    auto tmp10 = [&]
                    {
                        auto tmp11 = masked_load(in_ptr1 + static_cast<long>(x1 + (51328L*x0)), to_float_mask(tmp7));
                        auto tmp12 = masked_load(in_ptr2 + static_cast<long>(x1 + (51328L*x0)), to_float_mask(tmp7));
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = masked_load(in_ptr3 + static_cast<long>(x1 + (51328L*x0)), to_float_mask(tmp7));
                        auto tmp15 = tmp13 + tmp14;
                        return tmp15;
                    }
                    ;
                    auto tmp16 = decltype(tmp10())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp10(), to_float_mask(tmp7));
                    auto tmp17 = to_float_mask(tmp3);
                    auto tmp18 = decltype(tmp6)::blendv(tmp16, tmp6, tmp17);
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp21 = [&]
                    {
                        auto tmp22 = masked_load(in_ptr4 + static_cast<long>(x1 + (128L*x0)), to_float_mask(tmp3));
                        return tmp22;
                    }
                    ;
                    auto tmp23 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp3));
                    auto tmp24 = [&]
                    {
                        auto tmp25 = masked_load(in_ptr1 + static_cast<long>(x1 + (51328L*x0)), to_float_mask(tmp7));
                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>(x1 + (51328L*x0)), to_float_mask(tmp7));
                        auto tmp27 = tmp25 + tmp26;
                        auto tmp28 = masked_load(in_ptr3 + static_cast<long>(x1 + (51328L*x0)), to_float_mask(tmp7));
                        auto tmp29 = tmp27 + tmp28;
                        return tmp29;
                    }
                    ;
                    auto tmp30 = decltype(tmp24())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp24(), to_float_mask(tmp7));
                    auto tmp31 = decltype(tmp23)::blendv(tmp30, tmp23, tmp17);
                    auto tmp33 = at::vec::Vectorized<float>(tmp32);
                    auto tmp34 = tmp31 - tmp33;
                    auto tmp36 = static_cast<float>(128.0);
                    auto tmp37 = tmp35 / tmp36;
                    auto tmp38 = static_cast<float>(1e-06);
                    auto tmp39 = decltype(tmp37)(tmp37 + tmp38);
                    auto tmp40 = 1 / std::sqrt(tmp39);
                    auto tmp41 = at::vec::Vectorized<float>(tmp40);
                    auto tmp42 = tmp34 * tmp41;
                    auto tmp44 = tmp42 * tmp43;
                    auto tmp46 = tmp44 + tmp45;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp46.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp20);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp4 = out_ptr2[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-06);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
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
            tmp11.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                auto tmp16 = tmp14 + tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp18 = to_float_mask(tmp4);
                            auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                auto tmp27 = masked_load(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                                auto tmp28 = tmp26 + tmp27;
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp30 = decltype(tmp22)::blendv(tmp29, tmp22, tmp18);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp30);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
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
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp19 = in_ptr4[static_cast<long>(197L*x0)];
                auto tmp22 = in_ptr5[static_cast<long>(197L*x0)];
                auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp0 = static_cast<int>(0);
                auto tmp1 = tmp0 >= tmp0;
                auto tmp2 = static_cast<int>(1);
                auto tmp3 = tmp0 < tmp2;
                auto tmp4 = [&]
                {
                    auto tmp5 = masked_load(in_ptr0 + static_cast<long>(x1 + (256L*x0)), to_float_mask(tmp3));
                    return tmp5;
                }
                ;
                auto tmp6 = decltype(tmp4())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp4(), to_float_mask(tmp3));
                auto tmp7 = tmp0 >= tmp2;
                auto tmp8 = static_cast<int>(197);
                auto tmp9 = tmp0 < tmp8;
                auto tmp10 = [&]
                {
                    auto tmp11 = masked_load(in_ptr1 + static_cast<long>(x1 + (50432L*x0)), to_float_mask(tmp7));
                    auto tmp12 = masked_load(in_ptr2 + static_cast<long>(x1 + (50432L*x0)), to_float_mask(tmp7));
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = masked_load(in_ptr3 + static_cast<long>(x1 + (50432L*x0)), to_float_mask(tmp7));
                    auto tmp15 = tmp13 + tmp14;
                    return tmp15;
                }
                ;
                auto tmp16 = decltype(tmp10())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp10(), to_float_mask(tmp7));
                auto tmp17 = to_float_mask(tmp3);
                auto tmp18 = decltype(tmp6)::blendv(tmp16, tmp6, tmp17);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 - tmp20;
                auto tmp23 = static_cast<float>(256.0);
                auto tmp24 = tmp22 / tmp23;
                auto tmp25 = static_cast<float>(1e-06);
                auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                auto tmp27 = 1 / std::sqrt(tmp26);
                auto tmp28 = at::vec::Vectorized<float>(tmp27);
                auto tmp29 = tmp21 * tmp28;
                auto tmp31 = tmp29 * tmp30;
                auto tmp33 = tmp31 + tmp32;
                tmp33.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_mean_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(1000L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = c10::convert<int>(x0);
                auto tmp1 = static_cast<int>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<int>(8);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (1000L*x0)), to_float_mask(tmp4));
                    return tmp6;
                }
                ;
                auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<int>(16);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-8000L) + x1 + (1000L*x0)), to_float_mask(tmp8));
                    return tmp12;
                }
                ;
                auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                auto tmp14 = to_float_mask(tmp4);
                auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                auto tmp16 = c10::convert<int>(8L + x0);
                auto tmp17 = tmp16 >= tmp1;
                auto tmp18 = tmp16 < tmp3;
                auto tmp19 = [&]
                {
                    auto tmp20 = masked_load(in_ptr0 + static_cast<long>(8000L + x1 + (1000L*x0)), to_float_mask(tmp18));
                    return tmp20;
                }
                ;
                auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                auto tmp22 = tmp16 >= tmp3;
                auto tmp23 = tmp16 < tmp9;
                auto tmp24 = [&]
                {
                    auto tmp25 = masked_load(in_ptr1 + static_cast<long>(x1 + (1000L*x0)), to_float_mask(tmp22));
                    return tmp25;
                }
                ;
                auto tmp26 = decltype(tmp24())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp24(), to_float_mask(tmp22));
                auto tmp27 = to_float_mask(tmp18);
                auto tmp28 = decltype(tmp21)::blendv(tmp26, tmp21, tmp27);
                auto tmp29 = tmp15 + tmp28;
                auto tmp30 = static_cast<float>(2.0);
                auto tmp31 = at::vec::Vectorized<float>(tmp30);
                auto tmp32 = tmp29 / tmp31;
                tmp32.store(out_ptr0 + static_cast<long>(x1 + (1000L*x0)));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 128), (128, 128, 1))
    assert_size_stride(arg1_1, (1, 401, 128), (51328, 128, 1))
    assert_size_stride(arg2_1, (1, 1, 256), (256, 256, 1))
    assert_size_stride(arg3_1, (1, 197, 256), (50432, 256, 1))
    assert_size_stride(arg4_1, (128, 3, 12, 12), (432, 144, 12, 1))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (256, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg7_1, (256, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (384, 128), (128, 1))
    assert_size_stride(arg11_1, (384, ), (1, ))
    assert_size_stride(arg12_1, (128, 128), (128, 1))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (384, 128), (128, 1))
    assert_size_stride(arg17_1, (384, ), (1, ))
    assert_size_stride(arg18_1, (128, 384), (384, 1))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (768, 256), (256, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (256, 256), (256, 1))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (768, 256), (256, 1))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (256, 768), (768, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (768, 256), (256, 1))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (256, 256), (256, 1))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (768, 256), (256, 1))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (256, 768), (768, 1))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (768, 256), (256, 1))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (256, 256), (256, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (768, 256), (256, 1))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (256, 768), (768, 1))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (128, ), (1, ))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (256, 128), (128, 1))
    assert_size_stride(arg59_1, (256, ), (1, ))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (128, 256), (256, 1))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (256, ), (1, ))
    assert_size_stride(arg66_1, (256, 256), (256, 1))
    assert_size_stride(arg67_1, (256, ), (1, ))
    assert_size_stride(arg68_1, (256, 256), (256, 1))
    assert_size_stride(arg69_1, (256, ), (1, ))
    assert_size_stride(arg70_1, (256, 256), (256, 1))
    assert_size_stride(arg71_1, (256, ), (1, ))
    assert_size_stride(arg72_1, (256, 256), (256, 1))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, ), (1, ))
    assert_size_stride(arg76_1, (128, 256), (256, 1))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (128, 128), (128, 1))
    assert_size_stride(arg81_1, (128, ), (1, ))
    assert_size_stride(arg82_1, (128, 128), (128, 1))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, 128), (128, 1))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, 128), (128, 1))
    assert_size_stride(arg87_1, (128, ), (1, ))
    assert_size_stride(arg88_1, (128, ), (1, ))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (256, 128), (128, 1))
    assert_size_stride(arg91_1, (256, ), (1, ))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (384, 128), (128, 1))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (128, 128), (128, 1))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (384, 128), (128, 1))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (128, 384), (384, 1))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, ), (1, ))
    assert_size_stride(arg106_1, (768, 256), (256, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (256, 256), (256, 1))
    assert_size_stride(arg109_1, (256, ), (1, ))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (768, 256), (256, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (256, 768), (768, 1))
    assert_size_stride(arg115_1, (256, ), (1, ))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (256, ), (1, ))
    assert_size_stride(arg118_1, (768, 256), (256, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (256, 256), (256, 1))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (768, 256), (256, 1))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (256, 768), (768, 1))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (768, 256), (256, 1))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (256, 256), (256, 1))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (256, ), (1, ))
    assert_size_stride(arg135_1, (256, ), (1, ))
    assert_size_stride(arg136_1, (768, 256), (256, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (256, 768), (768, 1))
    assert_size_stride(arg139_1, (256, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (128, ), (1, ))
    assert_size_stride(arg142_1, (256, 128), (128, 1))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (128, 256), (256, 1))
    assert_size_stride(arg147_1, (128, ), (1, ))
    assert_size_stride(arg148_1, (256, ), (1, ))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (256, 256), (256, 1))
    assert_size_stride(arg151_1, (256, ), (1, ))
    assert_size_stride(arg152_1, (256, 256), (256, 1))
    assert_size_stride(arg153_1, (256, ), (1, ))
    assert_size_stride(arg154_1, (256, 256), (256, 1))
    assert_size_stride(arg155_1, (256, ), (1, ))
    assert_size_stride(arg156_1, (256, 256), (256, 1))
    assert_size_stride(arg157_1, (256, ), (1, ))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (256, ), (1, ))
    assert_size_stride(arg160_1, (128, 256), (256, 1))
    assert_size_stride(arg161_1, (128, ), (1, ))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (128, ), (1, ))
    assert_size_stride(arg164_1, (128, 128), (128, 1))
    assert_size_stride(arg165_1, (128, ), (1, ))
    assert_size_stride(arg166_1, (128, 128), (128, 1))
    assert_size_stride(arg167_1, (128, ), (1, ))
    assert_size_stride(arg168_1, (128, 128), (128, 1))
    assert_size_stride(arg169_1, (128, ), (1, ))
    assert_size_stride(arg170_1, (128, 128), (128, 1))
    assert_size_stride(arg171_1, (128, ), (1, ))
    assert_size_stride(arg172_1, (128, ), (1, ))
    assert_size_stride(arg173_1, (128, ), (1, ))
    assert_size_stride(arg174_1, (256, 128), (128, 1))
    assert_size_stride(arg175_1, (256, ), (1, ))
    assert_size_stride(arg176_1, (128, ), (1, ))
    assert_size_stride(arg177_1, (128, ), (1, ))
    assert_size_stride(arg178_1, (384, 128), (128, 1))
    assert_size_stride(arg179_1, (384, ), (1, ))
    assert_size_stride(arg180_1, (128, 128), (128, 1))
    assert_size_stride(arg181_1, (128, ), (1, ))
    assert_size_stride(arg182_1, (128, ), (1, ))
    assert_size_stride(arg183_1, (128, ), (1, ))
    assert_size_stride(arg184_1, (384, 128), (128, 1))
    assert_size_stride(arg185_1, (384, ), (1, ))
    assert_size_stride(arg186_1, (128, 384), (384, 1))
    assert_size_stride(arg187_1, (128, ), (1, ))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (256, ), (1, ))
    assert_size_stride(arg190_1, (768, 256), (256, 1))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (256, 256), (256, 1))
    assert_size_stride(arg193_1, (256, ), (1, ))
    assert_size_stride(arg194_1, (256, ), (1, ))
    assert_size_stride(arg195_1, (256, ), (1, ))
    assert_size_stride(arg196_1, (768, 256), (256, 1))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (256, 768), (768, 1))
    assert_size_stride(arg199_1, (256, ), (1, ))
    assert_size_stride(arg200_1, (256, ), (1, ))
    assert_size_stride(arg201_1, (256, ), (1, ))
    assert_size_stride(arg202_1, (768, 256), (256, 1))
    assert_size_stride(arg203_1, (768, ), (1, ))
    assert_size_stride(arg204_1, (256, 256), (256, 1))
    assert_size_stride(arg205_1, (256, ), (1, ))
    assert_size_stride(arg206_1, (256, ), (1, ))
    assert_size_stride(arg207_1, (256, ), (1, ))
    assert_size_stride(arg208_1, (768, 256), (256, 1))
    assert_size_stride(arg209_1, (768, ), (1, ))
    assert_size_stride(arg210_1, (256, 768), (768, 1))
    assert_size_stride(arg211_1, (256, ), (1, ))
    assert_size_stride(arg212_1, (256, ), (1, ))
    assert_size_stride(arg213_1, (256, ), (1, ))
    assert_size_stride(arg214_1, (768, 256), (256, 1))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (256, 256), (256, 1))
    assert_size_stride(arg217_1, (256, ), (1, ))
    assert_size_stride(arg218_1, (256, ), (1, ))
    assert_size_stride(arg219_1, (256, ), (1, ))
    assert_size_stride(arg220_1, (768, 256), (256, 1))
    assert_size_stride(arg221_1, (768, ), (1, ))
    assert_size_stride(arg222_1, (256, 768), (768, 1))
    assert_size_stride(arg223_1, (256, ), (1, ))
    assert_size_stride(arg224_1, (128, ), (1, ))
    assert_size_stride(arg225_1, (128, ), (1, ))
    assert_size_stride(arg226_1, (256, 128), (128, 1))
    assert_size_stride(arg227_1, (256, ), (1, ))
    assert_size_stride(arg228_1, (256, ), (1, ))
    assert_size_stride(arg229_1, (256, ), (1, ))
    assert_size_stride(arg230_1, (128, 256), (256, 1))
    assert_size_stride(arg231_1, (128, ), (1, ))
    assert_size_stride(arg232_1, (256, ), (1, ))
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (256, 256), (256, 1))
    assert_size_stride(arg235_1, (256, ), (1, ))
    assert_size_stride(arg236_1, (256, 256), (256, 1))
    assert_size_stride(arg237_1, (256, ), (1, ))
    assert_size_stride(arg238_1, (256, 256), (256, 1))
    assert_size_stride(arg239_1, (256, ), (1, ))
    assert_size_stride(arg240_1, (256, 256), (256, 1))
    assert_size_stride(arg241_1, (256, ), (1, ))
    assert_size_stride(arg242_1, (256, ), (1, ))
    assert_size_stride(arg243_1, (256, ), (1, ))
    assert_size_stride(arg244_1, (128, 256), (256, 1))
    assert_size_stride(arg245_1, (128, ), (1, ))
    assert_size_stride(arg246_1, (128, ), (1, ))
    assert_size_stride(arg247_1, (128, ), (1, ))
    assert_size_stride(arg248_1, (128, 128), (128, 1))
    assert_size_stride(arg249_1, (128, ), (1, ))
    assert_size_stride(arg250_1, (128, 128), (128, 1))
    assert_size_stride(arg251_1, (128, ), (1, ))
    assert_size_stride(arg252_1, (128, 128), (128, 1))
    assert_size_stride(arg253_1, (128, ), (1, ))
    assert_size_stride(arg254_1, (128, 128), (128, 1))
    assert_size_stride(arg255_1, (128, ), (1, ))
    assert_size_stride(arg256_1, (128, ), (1, ))
    assert_size_stride(arg257_1, (128, ), (1, ))
    assert_size_stride(arg258_1, (256, 128), (128, 1))
    assert_size_stride(arg259_1, (256, ), (1, ))
    assert_size_stride(arg260_1, (128, ), (1, ))
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (256, ), (1, ))
    assert_size_stride(arg263_1, (256, ), (1, ))
    assert_size_stride(arg264_1, (1000, 128), (128, 1))
    assert_size_stride(arg265_1, (1000, ), (1, ))
    assert_size_stride(arg266_1, (1000, 256), (256, 1))
    assert_size_stride(arg267_1, (1000, ), (1, ))
    assert_size_stride(arg268_1, (8, 3, 240, 240), (172800, 57600, 240, 1))
    buf0 = empty_strided((8, 3, 240, 240), (172800, 1, 720, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((128, 3, 12, 12), (432, 1, 36, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg268_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg4_1
    # Source Nodes: [l__mod___patch_embed_0_proj], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg5_1, stride=(12, 12), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 128, 20, 20), (51200, 1, 2560, 128))
    del arg5_1
    del buf0
    del buf1
    buf3 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_1(c_void_p(arg0_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg8_1
    del arg9_1
    buf7 = empty((3208, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_0_blocks_0___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg11_1, reinterpret_tensor(buf6, (3208, 128), (128, 1), 0), reinterpret_tensor(arg10_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf7)
    del arg10_1
    del arg11_1
    # Source Nodes: [x_3], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf8 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf7, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf7, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf7, (8, 4, 401, 32), (153984, 32, 384, 1), 256))
    buf9 = buf8[0]
    del buf8
    buf16 = reinterpret_tensor(buf6, (3208, 128), (128, 1), 0); del buf6  # reuse
    # Source Nodes: [x_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg13_1, reinterpret_tensor(buf9, (3208, 128), (128, 1), 0), reinterpret_tensor(arg12_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf16)
    del arg12_1
    del arg13_1
    buf17 = buf4; del buf4  # reuse
    buf18 = buf3; del buf3  # reuse
    buf21 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf23 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf26 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf28 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf45 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf47 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf50 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf52 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf57 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf59 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf62 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf64 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf24 = empty((8, 3, 224, 224), device='cpu', dtype=torch.float32)
    buf48 = empty((8, 3, 224, 224), device='cpu', dtype=torch.float32)
    buf60 = empty((8, 3, 224, 224), device='cpu', dtype=torch.float32)
    buf29 = buf24; del buf24  # reuse
    buf53 = buf48; del buf48  # reuse
    buf65 = buf60; del buf60  # reuse
    buf31 = empty((1, 1, 224, 1), device='cpu', dtype=torch.float32)
    buf33 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf35 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf38 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf40 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf36 = empty((8, 3, 224, 224), device='cpu', dtype=torch.float32)
    buf41 = buf36; del buf36  # reuse
    buf43 = empty((1, 1, 224, 1), device='cpu', dtype=torch.float32)
    buf55 = empty((1, 1, 224, 1), device='cpu', dtype=torch.float32)
    buf67 = empty((1, 1, 224, 1), device='cpu', dtype=torch.float32)
    buf68 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf69 = empty_strided((256, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_index_add_cat_convolution_mul_native_layer_norm_sub_2(c_void_p(buf29.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()))
    del arg268_1
    del arg6_1
    del buf21
    del buf23
    del buf26
    del buf28
    del buf29
    del buf31
    del buf33
    del buf35
    del buf38
    del buf40
    del buf41
    del buf43
    del buf45
    del buf47
    del buf50
    del buf52
    del buf53
    del buf55
    del buf57
    del buf59
    del buf62
    del buf64
    del buf65
    del buf67
    # Source Nodes: [l__mod___patch_embed_1_proj, x__5], Original ATen: [aten.add, aten.convolution, aten.mul]
    buf70 = extern_kernels.convolution(buf68, buf69, arg7_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf70, (8, 256, 14, 14), (50176, 1, 3584, 256))
    del arg7_1
    del buf68
    del buf69
    buf71 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf72 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf74 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_3(c_void_p(arg2_1.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()))
    del arg20_1
    del arg21_1
    buf75 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg23_1, reinterpret_tensor(buf74, (1576, 256), (256, 1), 0), reinterpret_tensor(arg22_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf75)
    del arg22_1
    del arg23_1
    # Source Nodes: [x_15], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf76 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf75, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf75, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf75, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf77 = buf76[0]
    del buf76
    buf84 = reinterpret_tensor(buf74, (1576, 256), (256, 1), 0); del buf74  # reuse
    # Source Nodes: [x_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg25_1, reinterpret_tensor(buf77, (1576, 256), (256, 1), 0), reinterpret_tensor(arg24_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf84)
    del arg24_1
    del arg25_1
    buf85 = buf72; del buf72  # reuse
    buf86 = buf71; del buf71  # reuse
    buf88 = reinterpret_tensor(buf77, (8, 197, 256), (50432, 256, 1), 0); del buf77  # reuse
    cpp_fused_add_cat_native_layer_norm_4(c_void_p(arg2_1.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()))
    del arg26_1
    del arg27_1
    buf89 = buf75; del buf75  # reuse
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg29_1, reinterpret_tensor(buf88, (1576, 256), (256, 1), 0), reinterpret_tensor(arg28_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf89)
    del arg28_1
    del arg29_1
    buf90 = reinterpret_tensor(buf89, (8, 197, 768), (151296, 768, 1), 0); del buf89  # reuse
    cpp_fused_gelu_5(c_void_p(buf90.data_ptr()))
    buf91 = reinterpret_tensor(buf88, (1576, 256), (256, 1), 0); del buf88  # reuse
    # Source Nodes: [x_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg31_1, reinterpret_tensor(buf90, (1576, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf91)
    del arg30_1
    del arg31_1
    buf92 = reinterpret_tensor(buf91, (8, 197, 256), (50432, 256, 1), 0); del buf91  # reuse
    buf93 = buf86; del buf86  # reuse
    buf94 = buf85; del buf85  # reuse
    buf96 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_6(c_void_p(buf92.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()))
    del arg2_1
    del arg32_1
    del arg33_1
    del arg3_1
    del buf70
    buf97 = reinterpret_tensor(buf90, (1576, 768), (768, 1), 0); del buf90  # reuse
    # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg35_1, reinterpret_tensor(buf96, (1576, 256), (256, 1), 0), reinterpret_tensor(arg34_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf97)
    del arg34_1
    del arg35_1
    # Source Nodes: [x_27], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf98 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf97, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf97, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf97, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf99 = buf98[0]
    del buf98
    buf106 = reinterpret_tensor(buf96, (1576, 256), (256, 1), 0); del buf96  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg37_1, reinterpret_tensor(buf99, (1576, 256), (256, 1), 0), reinterpret_tensor(arg36_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf106)
    del arg36_1
    del arg37_1
    buf107 = buf94; del buf94  # reuse
    buf108 = buf93; del buf93  # reuse
    buf110 = reinterpret_tensor(buf99, (8, 197, 256), (50432, 256, 1), 0); del buf99  # reuse
    cpp_fused_add_native_layer_norm_7(c_void_p(buf92.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf110.data_ptr()))
    del arg38_1
    del arg39_1
    buf111 = buf97; del buf97  # reuse
    # Source Nodes: [x_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg41_1, reinterpret_tensor(buf110, (1576, 256), (256, 1), 0), reinterpret_tensor(arg40_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf111)
    del arg40_1
    del arg41_1
    buf112 = reinterpret_tensor(buf111, (8, 197, 768), (151296, 768, 1), 0); del buf111  # reuse
    cpp_fused_gelu_8(c_void_p(buf112.data_ptr()))
    buf113 = reinterpret_tensor(buf110, (1576, 256), (256, 1), 0); del buf110  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg43_1, reinterpret_tensor(buf112, (1576, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf113)
    del arg42_1
    del arg43_1
    buf114 = buf108; del buf108  # reuse
    buf115 = buf107; del buf107  # reuse
    buf117 = reinterpret_tensor(buf84, (8, 197, 256), (50432, 256, 1), 0); del buf84  # reuse
    cpp_fused_add_native_layer_norm_9(c_void_p(buf92.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()))
    del arg44_1
    del arg45_1
    buf118 = reinterpret_tensor(buf112, (1576, 768), (768, 1), 0); del buf112  # reuse
    # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg47_1, reinterpret_tensor(buf117, (1576, 256), (256, 1), 0), reinterpret_tensor(arg46_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf118)
    del arg46_1
    del arg47_1
    # Source Nodes: [x_39], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf119 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf118, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf118, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf118, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf120 = buf119[0]
    del buf119
    buf127 = reinterpret_tensor(buf117, (1576, 256), (256, 1), 0); del buf117  # reuse
    # Source Nodes: [x_41], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg49_1, reinterpret_tensor(buf120, (1576, 256), (256, 1), 0), reinterpret_tensor(arg48_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf127)
    del arg48_1
    del arg49_1
    buf128 = buf115; del buf115  # reuse
    buf129 = buf114; del buf114  # reuse
    buf131 = reinterpret_tensor(buf9, (8, 401, 128), (51328, 128, 1), 0); del buf9  # reuse
    cpp_fused_add_cat_native_layer_norm_10(c_void_p(buf92.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf131.data_ptr()))
    del arg14_1
    del arg15_1
    buf132 = buf7; del buf7  # reuse
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg17_1, reinterpret_tensor(buf131, (3208, 128), (128, 1), 0), reinterpret_tensor(arg16_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf132)
    del arg16_1
    del arg17_1
    buf133 = reinterpret_tensor(buf132, (8, 401, 384), (153984, 384, 1), 0); del buf132  # reuse
    cpp_fused_gelu_11(c_void_p(buf133.data_ptr()))
    buf134 = reinterpret_tensor(buf131, (3208, 128), (128, 1), 0); del buf131  # reuse
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg19_1, reinterpret_tensor(buf133, (3208, 384), (384, 1), 0), reinterpret_tensor(arg18_1, (384, 128), (1, 384), 0), alpha=1, beta=1, out=buf134)
    del arg18_1
    del arg19_1
    buf135 = reinterpret_tensor(buf134, (8, 401, 128), (51328, 128, 1), 0); del buf134  # reuse
    buf136 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf137 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf139 = reinterpret_tensor(buf120, (8, 197, 256), (50432, 256, 1), 0); del buf120  # reuse
    cpp_fused_add_cat_native_layer_norm_12(c_void_p(buf135.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()))
    del arg0_1
    del arg1_1
    del arg50_1
    del arg51_1
    del buf2
    buf140 = buf118; del buf118  # reuse
    # Source Nodes: [x_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg53_1, reinterpret_tensor(buf139, (1576, 256), (256, 1), 0), reinterpret_tensor(arg52_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf140)
    del arg52_1
    del arg53_1
    buf141 = reinterpret_tensor(buf140, (8, 197, 768), (151296, 768, 1), 0); del buf140  # reuse
    cpp_fused_gelu_13(c_void_p(buf141.data_ptr()))
    buf142 = reinterpret_tensor(buf139, (1576, 256), (256, 1), 0); del buf139  # reuse
    # Source Nodes: [x_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg55_1, reinterpret_tensor(buf141, (1576, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf142)
    del arg54_1
    del arg55_1
    buf143 = reinterpret_tensor(buf142, (8, 197, 256), (50432, 256, 1), 0); del buf142  # reuse
    buf147 = empty_strided((8, 1, 128), (128, 1024, 1), device='cpu', dtype=torch.float32)
    buf148 = reinterpret_tensor(buf147, (8, 1, 128), (128, 128, 1), 0); del buf147  # reuse
    cpp_fused_add_gelu_native_layer_norm_14(c_void_p(buf143.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()))
    del arg56_1
    del arg57_1
    del buf106
    buf149 = empty((8, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_projs_0_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg59_1, reinterpret_tensor(buf148, (8, 128), (128, 1), 0), reinterpret_tensor(arg58_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf149)
    del arg58_1
    del arg59_1
    buf150 = buf129; del buf129  # reuse
    buf151 = buf128; del buf128  # reuse
    buf153 = buf92; del buf92  # reuse
    cpp_fused_cat_native_layer_norm_15(c_void_p(buf149.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()))
    del arg64_1
    del arg65_1
    buf154 = empty((8, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_fusion_0_attn_wq], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf153, (8, 256), (50432, 1), 0), reinterpret_tensor(arg66_1, (256, 256), (1, 256), 0), out=buf154)
    del arg66_1
    buf156 = reinterpret_tensor(buf154, (8, 1, 256), (256, 256, 1), 0); del buf154  # reuse
    cpp_fused_add_16(c_void_p(buf156.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg67_1
    buf155 = buf127; del buf127  # reuse
    # Source Nodes: [l__mod___blocks_0_fusion_0_attn_wk], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg69_1, reinterpret_tensor(buf153, (1576, 256), (256, 1), 0), reinterpret_tensor(arg68_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf155)
    del arg68_1
    del arg69_1
    buf157 = reinterpret_tensor(buf113, (8, 4, 64, 197), (50432, 12608, 197, 1), 0); del buf113  # reuse
    cpp_fused_clone_17(c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()))
    buf158 = empty((32, 1, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf156, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf157, (32, 64, 197), (12608, 197, 1), 0), out=buf158)
    buf159 = empty_strided((8, 4, 1, 1), (4, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf160 = reinterpret_tensor(buf158, (8, 4, 1, 197), (788, 197, 6304, 1), 0); del buf158  # reuse
    buf161 = empty_strided((8, 4, 1, 1), (4, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf163 = reinterpret_tensor(buf160, (8, 4, 1, 197), (788, 197, 197, 1), 0); del buf160  # reuse
    cpp_fused__softmax_mul_18(c_void_p(buf163.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()))
    buf162 = reinterpret_tensor(buf157, (1576, 256), (256, 1), 0); del buf157  # reuse
    # Source Nodes: [l__mod___blocks_0_fusion_0_attn_wv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg71_1, reinterpret_tensor(buf153, (1576, 256), (256, 1), 0), reinterpret_tensor(arg70_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf162)
    del arg70_1
    del arg71_1
    buf164 = reinterpret_tensor(buf153, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf153  # reuse
    cpp_fused_clone_19(c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()))
    buf165 = reinterpret_tensor(buf156, (32, 1, 64), (64, 64, 1), 0); del buf156  # reuse
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf163, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf164, (32, 197, 64), (12608, 64, 1), 0), out=buf165)
    buf166 = empty((8, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_52], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg73_1, reinterpret_tensor(buf165, (8, 256), (256, 1), 0), reinterpret_tensor(arg72_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf166)
    del arg72_1
    del arg73_1
    buf144 = buf137; del buf137  # reuse
    buf145 = buf136; del buf136  # reuse
    buf167 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf168 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf170 = reinterpret_tensor(buf165, (8, 1, 256), (256, 2048, 1), 0); del buf165  # reuse
    buf193 = empty_strided((8, 1, 256), (256, 2048, 1), device='cpu', dtype=torch.float32)
    buf171 = reinterpret_tensor(buf170, (8, 1, 256), (256, 256, 1), 0); del buf170  # reuse
    cpp_fused_add_gelu_native_layer_norm_20(c_void_p(buf171.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf193.data_ptr()))
    del arg60_1
    del arg61_1
    del arg74_1
    del arg75_1
    del buf149
    buf172 = reinterpret_tensor(buf148, (8, 128), (128, 1), 0); del buf148  # reuse
    # Source Nodes: [l__mod___blocks_0_projs_1_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg63_1, reinterpret_tensor(buf171, (8, 256), (256, 1), 0), reinterpret_tensor(arg62_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf172)
    del arg62_1
    del arg63_1
    buf194 = reinterpret_tensor(buf193, (8, 1, 256), (256, 256, 1), 0); del buf193  # reuse
    cpp_fused_gelu_21(c_void_p(buf194.data_ptr()))
    buf195 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [reverted_proj_cls_token], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg77_1, reinterpret_tensor(buf194, (8, 256), (256, 1), 0), reinterpret_tensor(arg76_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf195)
    del arg76_1
    del arg77_1
    buf173 = buf18; del buf18  # reuse
    buf174 = buf17; del buf17  # reuse
    buf196 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf197 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf176 = reinterpret_tensor(buf16, (8, 401, 128), (51328, 128, 1), 0); del buf16  # reuse
    buf199 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_22(c_void_p(buf172.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf199.data_ptr()))
    del arg78_1
    del arg79_1
    del arg92_1
    del arg93_1
    buf177 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_fusion_1_attn_wq], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf176, (8, 128), (51328, 1), 0), reinterpret_tensor(arg80_1, (128, 128), (1, 128), 0), out=buf177)
    del arg80_1
    buf178 = empty((3208, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_fusion_1_attn_wk], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg83_1, reinterpret_tensor(buf176, (3208, 128), (128, 1), 0), reinterpret_tensor(arg82_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf178)
    del arg82_1
    del arg83_1
    buf179 = reinterpret_tensor(buf177, (8, 1, 128), (128, 128, 1), 0); del buf177  # reuse
    buf180 = empty((8, 4, 32, 401), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_23(c_void_p(buf179.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()))
    del arg81_1
    buf181 = empty((32, 1, 401), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf179, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf180, (32, 32, 401), (12832, 401, 1), 0), out=buf181)
    buf182 = buf161; del buf161  # reuse
    buf183 = reinterpret_tensor(buf181, (8, 4, 1, 401), (1604, 401, 12832, 1), 0); del buf181  # reuse
    buf184 = buf159; del buf159  # reuse
    cpp_fused__softmax_mul_24(c_void_p(buf183.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()))
    buf185 = reinterpret_tensor(buf180, (3208, 128), (128, 1), 0); del buf180  # reuse
    # Source Nodes: [l__mod___blocks_0_fusion_1_attn_wv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf176, (3208, 128), (128, 1), 0), reinterpret_tensor(arg84_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf185)
    del arg84_1
    del arg85_1
    buf186 = reinterpret_tensor(buf183, (8, 4, 1, 401), (1604, 401, 401, 1), 0); del buf183  # reuse
    buf187 = reinterpret_tensor(buf176, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf176  # reuse
    cpp_fused__softmax_clone_25(c_void_p(buf186.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()))
    buf188 = reinterpret_tensor(buf179, (32, 1, 32), (32, 32, 1), 0); del buf179  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf186, (32, 1, 401), (401, 0, 1), 0), reinterpret_tensor(buf187, (32, 401, 32), (12832, 32, 1), 0), out=buf188)
    buf189 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg87_1, reinterpret_tensor(buf188, (8, 128), (128, 1), 0), reinterpret_tensor(arg86_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf189)
    del arg86_1
    del arg87_1
    buf200 = reinterpret_tensor(buf133, (3208, 384), (384, 1), 0); del buf133  # reuse
    # Source Nodes: [getattr_l__mod___blocks_1_blocks_0___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg95_1, reinterpret_tensor(buf199, (3208, 128), (128, 1), 0), reinterpret_tensor(arg94_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf200)
    del arg94_1
    del arg95_1
    # Source Nodes: [x_59], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf201 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf200, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf200, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf200, (8, 4, 401, 32), (153984, 32, 384, 1), 256))
    buf202 = buf201[0]
    del buf201
    buf209 = reinterpret_tensor(buf199, (3208, 128), (128, 1), 0); del buf199  # reuse
    # Source Nodes: [x_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg97_1, reinterpret_tensor(buf202, (3208, 128), (128, 1), 0), reinterpret_tensor(arg96_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf209)
    del arg96_1
    del arg97_1
    buf210 = buf197; del buf197  # reuse
    buf211 = buf196; del buf196  # reuse
    buf276 = reinterpret_tensor(buf202, (8, 401, 128), (51328, 128, 1), 0); del buf202  # reuse
    cpp_fused_add_cat_native_layer_norm_26(c_void_p(buf195.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf276.data_ptr()))
    del arg98_1
    del arg99_1
    buf277 = buf200; del buf200  # reuse
    # Source Nodes: [x_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg101_1, reinterpret_tensor(buf276, (3208, 128), (128, 1), 0), reinterpret_tensor(arg100_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf277)
    del arg100_1
    del arg101_1
    buf278 = reinterpret_tensor(buf277, (8, 401, 384), (153984, 384, 1), 0); del buf277  # reuse
    cpp_fused_gelu_27(c_void_p(buf278.data_ptr()))
    buf279 = reinterpret_tensor(buf276, (3208, 128), (128, 1), 0); del buf276  # reuse
    # Source Nodes: [x_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg103_1, reinterpret_tensor(buf278, (3208, 384), (384, 1), 0), reinterpret_tensor(arg102_1, (384, 128), (1, 384), 0), alpha=1, beta=1, out=buf279)
    del arg102_1
    del arg103_1
    buf190 = buf168; del buf168  # reuse
    buf191 = buf167; del buf167  # reuse
    buf280 = buf145; del buf145  # reuse
    buf281 = buf144; del buf144  # reuse
    buf213 = reinterpret_tensor(buf188, (8, 1, 128), (128, 1024, 1), 0); del buf188  # reuse
    buf290 = empty_strided((8, 1, 128), (128, 1024, 1), device='cpu', dtype=torch.float32)
    buf214 = reinterpret_tensor(buf213, (8, 1, 128), (128, 128, 1), 0); del buf213  # reuse
    cpp_fused_add_gelu_native_layer_norm_28(c_void_p(buf214.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf290.data_ptr()))
    del arg140_1
    del arg141_1
    del arg88_1
    del arg89_1
    buf215 = reinterpret_tensor(buf194, (8, 256), (256, 1), 0); del buf194  # reuse
    # Source Nodes: [reverted_proj_cls_token_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf214, (8, 128), (128, 1), 0), reinterpret_tensor(arg90_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf215)
    del arg90_1
    del arg91_1
    buf216 = buf151; del buf151  # reuse
    buf217 = buf150; del buf150  # reuse
    buf219 = reinterpret_tensor(buf164, (8, 197, 256), (50432, 256, 1), 0); del buf164  # reuse
    cpp_fused_cat_native_layer_norm_29(c_void_p(buf215.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()))
    del arg104_1
    del arg105_1
    buf220 = reinterpret_tensor(buf141, (1576, 768), (768, 1), 0); del buf141  # reuse
    # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg107_1, reinterpret_tensor(buf219, (1576, 256), (256, 1), 0), reinterpret_tensor(arg106_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf220)
    del arg106_1
    del arg107_1
    # Source Nodes: [x_71], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf221 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf220, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf220, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf220, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf222 = buf221[0]
    del buf221
    buf229 = reinterpret_tensor(buf219, (1576, 256), (256, 1), 0); del buf219  # reuse
    # Source Nodes: [x_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg109_1, reinterpret_tensor(buf222, (1576, 256), (256, 1), 0), reinterpret_tensor(arg108_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf229)
    del arg108_1
    del arg109_1
    buf230 = buf217; del buf217  # reuse
    buf231 = buf216; del buf216  # reuse
    buf233 = reinterpret_tensor(buf222, (8, 197, 256), (50432, 256, 1), 0); del buf222  # reuse
    cpp_fused_add_cat_native_layer_norm_30(c_void_p(buf215.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf233.data_ptr()))
    del arg110_1
    del arg111_1
    buf234 = buf220; del buf220  # reuse
    # Source Nodes: [x_76], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg113_1, reinterpret_tensor(buf233, (1576, 256), (256, 1), 0), reinterpret_tensor(arg112_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf234)
    del arg112_1
    del arg113_1
    buf235 = reinterpret_tensor(buf234, (8, 197, 768), (151296, 768, 1), 0); del buf234  # reuse
    cpp_fused_gelu_31(c_void_p(buf235.data_ptr()))
    buf236 = reinterpret_tensor(buf233, (1576, 256), (256, 1), 0); del buf233  # reuse
    # Source Nodes: [x_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg115_1, reinterpret_tensor(buf235, (1576, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf236)
    del arg114_1
    del arg115_1
    buf237 = buf231; del buf231  # reuse
    buf238 = buf230; del buf230  # reuse
    buf240 = reinterpret_tensor(buf162, (8, 197, 256), (50432, 256, 1), 0); del buf162  # reuse
    cpp_fused_add_cat_native_layer_norm_32(c_void_p(buf215.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()))
    del arg116_1
    del arg117_1
    buf241 = reinterpret_tensor(buf235, (1576, 768), (768, 1), 0); del buf235  # reuse
    # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg119_1, reinterpret_tensor(buf240, (1576, 256), (256, 1), 0), reinterpret_tensor(arg118_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf241)
    del arg118_1
    del arg119_1
    # Source Nodes: [x_83], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf242 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf241, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf241, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf241, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf243 = buf242[0]
    del buf242
    buf250 = reinterpret_tensor(buf240, (1576, 256), (256, 1), 0); del buf240  # reuse
    # Source Nodes: [x_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg121_1, reinterpret_tensor(buf243, (1576, 256), (256, 1), 0), reinterpret_tensor(arg120_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf250)
    del arg120_1
    del arg121_1
    buf251 = reinterpret_tensor(buf250, (8, 197, 256), (50432, 256, 1), 0); del buf250  # reuse
    buf252 = buf238; del buf238  # reuse
    buf253 = buf237; del buf237  # reuse
    buf255 = reinterpret_tensor(buf243, (8, 197, 256), (50432, 256, 1), 0); del buf243  # reuse
    cpp_fused_add_cat_native_layer_norm_33(c_void_p(buf251.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf255.data_ptr()))
    del arg122_1
    del arg123_1
    buf256 = buf241; del buf241  # reuse
    # Source Nodes: [x_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg125_1, reinterpret_tensor(buf255, (1576, 256), (256, 1), 0), reinterpret_tensor(arg124_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf256)
    del arg124_1
    del arg125_1
    buf257 = reinterpret_tensor(buf256, (8, 197, 768), (151296, 768, 1), 0); del buf256  # reuse
    cpp_fused_gelu_34(c_void_p(buf257.data_ptr()))
    buf258 = reinterpret_tensor(buf255, (1576, 256), (256, 1), 0); del buf255  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg127_1, reinterpret_tensor(buf257, (1576, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf258)
    del arg126_1
    del arg127_1
    buf259 = buf253; del buf253  # reuse
    buf260 = buf252; del buf252  # reuse
    buf262 = reinterpret_tensor(buf236, (8, 197, 256), (50432, 256, 1), 0); del buf236  # reuse
    cpp_fused_add_native_layer_norm_35(c_void_p(buf251.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf262.data_ptr()))
    del arg128_1
    del arg129_1
    buf263 = reinterpret_tensor(buf257, (1576, 768), (768, 1), 0); del buf257  # reuse
    # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg131_1, reinterpret_tensor(buf262, (1576, 256), (256, 1), 0), reinterpret_tensor(arg130_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf263)
    del arg130_1
    del arg131_1
    # Source Nodes: [x_95], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf264 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf263, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf263, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf263, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf265 = buf264[0]
    del buf264
    buf272 = reinterpret_tensor(buf262, (1576, 256), (256, 1), 0); del buf262  # reuse
    # Source Nodes: [x_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg133_1, reinterpret_tensor(buf265, (1576, 256), (256, 1), 0), reinterpret_tensor(arg132_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf272)
    del arg132_1
    del arg133_1
    buf273 = buf260; del buf260  # reuse
    buf274 = buf259; del buf259  # reuse
    buf283 = reinterpret_tensor(buf265, (8, 197, 256), (50432, 256, 1), 0); del buf265  # reuse
    cpp_fused_add_native_layer_norm_36(c_void_p(buf251.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf283.data_ptr()))
    del arg134_1
    del arg135_1
    buf284 = buf263; del buf263  # reuse
    # Source Nodes: [x_100], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg137_1, reinterpret_tensor(buf283, (1576, 256), (256, 1), 0), reinterpret_tensor(arg136_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf284)
    del arg136_1
    del arg137_1
    buf285 = reinterpret_tensor(buf284, (8, 197, 768), (151296, 768, 1), 0); del buf284  # reuse
    cpp_fused_gelu_37(c_void_p(buf285.data_ptr()))
    buf286 = reinterpret_tensor(buf283, (1576, 256), (256, 1), 0); del buf283  # reuse
    # Source Nodes: [x_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg139_1, reinterpret_tensor(buf285, (1576, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf286)
    del arg138_1
    del arg139_1
    buf287 = buf281; del buf281  # reuse
    buf288 = buf280; del buf280  # reuse
    buf291 = reinterpret_tensor(buf290, (8, 1, 128), (128, 128, 1), 0); del buf290  # reuse
    cpp_fused_gelu_native_layer_norm_38(c_void_p(buf291.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    buf292 = buf215; del buf215  # reuse
    # Source Nodes: [l__mod___blocks_1_projs_0_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, reinterpret_tensor(buf291, (8, 128), (128, 1), 0), reinterpret_tensor(arg142_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf292)
    del arg142_1
    del arg143_1
    buf314 = reinterpret_tensor(buf171, (8, 1, 256), (256, 2048, 1), 0); del buf171  # reuse
    buf315 = reinterpret_tensor(buf314, (8, 1, 256), (256, 256, 1), 0); del buf314  # reuse
    cpp_fused_gelu_native_layer_norm_39(c_void_p(buf315.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg145_1.data_ptr()))
    del arg144_1
    del arg145_1
    buf316 = reinterpret_tensor(buf291, (8, 128), (128, 1), 0); del buf291  # reuse
    # Source Nodes: [l__mod___blocks_1_projs_1_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg147_1, reinterpret_tensor(buf315, (8, 256), (256, 1), 0), reinterpret_tensor(arg146_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf316)
    del arg146_1
    del arg147_1
    buf317 = reinterpret_tensor(buf187, (8, 401, 128), (51328, 128, 1), 0); del buf187  # reuse
    buf318 = buf211; del buf211  # reuse
    buf319 = buf210; del buf210  # reuse
    buf321 = reinterpret_tensor(buf185, (8, 401, 128), (51328, 128, 1), 0); del buf185  # reuse
    cpp_fused_cat_native_layer_norm_40(c_void_p(buf316.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf321.data_ptr()))
    del arg162_1
    del arg163_1
    buf322 = buf316; del buf316  # reuse
    # Source Nodes: [l__mod___blocks_1_fusion_1_attn_wq], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf321, (8, 128), (51328, 1), 0), reinterpret_tensor(arg164_1, (128, 128), (1, 128), 0), out=buf322)
    del arg164_1
    buf324 = reinterpret_tensor(buf322, (8, 1, 128), (128, 128, 1), 0); del buf322  # reuse
    cpp_fused_add_41(c_void_p(buf324.data_ptr()), c_void_p(arg165_1.data_ptr()))
    del arg165_1
    buf323 = buf178; del buf178  # reuse
    # Source Nodes: [l__mod___blocks_1_fusion_1_attn_wk], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg167_1, reinterpret_tensor(buf321, (3208, 128), (128, 1), 0), reinterpret_tensor(arg166_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf323)
    del arg166_1
    del arg167_1
    buf325 = empty((8, 4, 32, 401), device='cpu', dtype=torch.float32)
    cpp_fused_clone_42(c_void_p(buf323.data_ptr()), c_void_p(buf325.data_ptr()))
    del buf323
    buf326 = reinterpret_tensor(buf186, (32, 1, 401), (401, 401, 1), 0); del buf186  # reuse
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf324, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf325, (32, 32, 401), (12832, 401, 1), 0), out=buf326)
    buf327 = buf184; del buf184  # reuse
    buf328 = reinterpret_tensor(buf326, (8, 4, 1, 401), (1604, 401, 12832, 1), 0); del buf326  # reuse
    buf329 = buf182; del buf182  # reuse
    buf331 = reinterpret_tensor(buf328, (8, 4, 1, 401), (1604, 401, 401, 1), 0); del buf328  # reuse
    cpp_fused__softmax_mul_43(c_void_p(buf331.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf329.data_ptr()))
    buf330 = reinterpret_tensor(buf325, (3208, 128), (128, 1), 0); del buf325  # reuse
    # Source Nodes: [l__mod___blocks_1_fusion_1_attn_wv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg169_1, reinterpret_tensor(buf321, (3208, 128), (128, 1), 0), reinterpret_tensor(arg168_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf330)
    del arg168_1
    del arg169_1
    buf332 = reinterpret_tensor(buf321, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf321  # reuse
    cpp_fused_clone_44(c_void_p(buf330.data_ptr()), c_void_p(buf332.data_ptr()))
    del buf330
    buf333 = reinterpret_tensor(buf324, (32, 1, 32), (32, 32, 1), 0); del buf324  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf331, (32, 1, 401), (401, 0, 1), 0), reinterpret_tensor(buf332, (32, 401, 32), (12832, 32, 1), 0), out=buf333)
    buf334 = reinterpret_tensor(buf214, (8, 128), (128, 1), 0); del buf214  # reuse
    # Source Nodes: [x_112], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg171_1, reinterpret_tensor(buf333, (8, 128), (128, 1), 0), reinterpret_tensor(arg170_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf334)
    del arg170_1
    del arg171_1
    buf335 = buf288; del buf288  # reuse
    buf336 = buf287; del buf287  # reuse
    buf359 = reinterpret_tensor(buf333, (8, 1, 128), (128, 1024, 1), 0); del buf333  # reuse
    buf360 = reinterpret_tensor(buf359, (8, 1, 128), (128, 128, 1), 0); del buf359  # reuse
    cpp_fused_add_gelu_native_layer_norm_45(c_void_p(buf360.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    del arg172_1
    del arg173_1
    buf361 = reinterpret_tensor(buf315, (8, 256), (256, 1), 0); del buf315  # reuse
    # Source Nodes: [reverted_proj_cls_token_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg175_1, reinterpret_tensor(buf360, (8, 128), (128, 1), 0), reinterpret_tensor(arg174_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf361)
    del arg174_1
    del arg175_1
    buf293 = reinterpret_tensor(buf229, (8, 197, 256), (50432, 256, 1), 0); del buf229  # reuse
    buf362 = buf143; del buf143  # reuse
    buf294 = buf274; del buf274  # reuse
    buf295 = buf273; del buf273  # reuse
    buf297 = reinterpret_tensor(buf155, (8, 197, 256), (50432, 256, 1), 0); del buf155  # reuse
    cpp_fused_cat_native_layer_norm_46(c_void_p(buf292.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()))
    del arg148_1
    del arg149_1
    del buf251
    del buf258
    buf298 = buf361; del buf361  # reuse
    # Source Nodes: [l__mod___blocks_1_fusion_0_attn_wq], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (8, 256), (50432, 1), 0), reinterpret_tensor(arg150_1, (256, 256), (1, 256), 0), out=buf298)
    del arg150_1
    buf299 = buf286; del buf286  # reuse
    # Source Nodes: [l__mod___blocks_1_fusion_0_attn_wk], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg153_1, reinterpret_tensor(buf297, (1576, 256), (256, 1), 0), reinterpret_tensor(arg152_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf299)
    del arg152_1
    del arg153_1
    buf300 = reinterpret_tensor(buf298, (8, 1, 256), (256, 256, 1), 0); del buf298  # reuse
    buf301 = reinterpret_tensor(buf272, (8, 4, 64, 197), (50432, 12608, 197, 1), 0); del buf272  # reuse
    cpp_fused_add_clone_47(c_void_p(buf300.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf301.data_ptr()))
    del arg151_1
    del buf299
    buf302 = reinterpret_tensor(buf163, (32, 1, 197), (197, 197, 1), 0); del buf163  # reuse
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf300, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf301, (32, 64, 197), (12608, 197, 1), 0), out=buf302)
    buf303 = buf329; del buf329  # reuse
    buf304 = reinterpret_tensor(buf302, (8, 4, 1, 197), (788, 197, 6304, 1), 0); del buf302  # reuse
    buf305 = buf327; del buf327  # reuse
    cpp_fused__softmax_mul_48(c_void_p(buf304.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf305.data_ptr()))
    buf306 = reinterpret_tensor(buf301, (1576, 256), (256, 1), 0); del buf301  # reuse
    # Source Nodes: [l__mod___blocks_1_fusion_0_attn_wv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg155_1, reinterpret_tensor(buf297, (1576, 256), (256, 1), 0), reinterpret_tensor(arg154_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf306)
    del arg154_1
    del arg155_1
    buf307 = reinterpret_tensor(buf304, (8, 4, 1, 197), (788, 197, 197, 1), 0); del buf304  # reuse
    buf308 = reinterpret_tensor(buf297, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf297  # reuse
    cpp_fused__softmax_clone_49(c_void_p(buf307.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf308.data_ptr()))
    buf309 = reinterpret_tensor(buf300, (32, 1, 64), (64, 64, 1), 0); del buf300  # reuse
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf307, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf308, (32, 197, 64), (12608, 64, 1), 0), out=buf309)
    buf310 = buf292; del buf292  # reuse
    # Source Nodes: [x_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg157_1, reinterpret_tensor(buf309, (8, 256), (256, 1), 0), reinterpret_tensor(arg156_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf310)
    del arg156_1
    del arg157_1
    buf311 = buf336; del buf336  # reuse
    buf312 = buf335; del buf335  # reuse
    buf338 = reinterpret_tensor(buf309, (8, 1, 256), (256, 2048, 1), 0); del buf309  # reuse
    buf339 = reinterpret_tensor(buf338, (8, 1, 256), (256, 256, 1), 0); del buf338  # reuse
    cpp_fused_add_gelu_native_layer_norm_50(c_void_p(buf339.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    del arg158_1
    del arg159_1
    buf340 = reinterpret_tensor(buf360, (8, 128), (128, 1), 0); del buf360  # reuse
    # Source Nodes: [reverted_proj_cls_token_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg161_1, reinterpret_tensor(buf339, (8, 256), (256, 1), 0), reinterpret_tensor(arg160_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf340)
    del arg160_1
    del arg161_1
    buf341 = buf317; del buf317  # reuse
    buf342 = buf319; del buf319  # reuse
    buf343 = buf318; del buf318  # reuse
    buf345 = reinterpret_tensor(buf332, (8, 401, 128), (51328, 128, 1), 0); del buf332  # reuse
    cpp_fused_cat_native_layer_norm_51(c_void_p(buf340.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf345.data_ptr()))
    del arg176_1
    del arg177_1
    buf346 = reinterpret_tensor(buf278, (3208, 384), (384, 1), 0); del buf278  # reuse
    # Source Nodes: [getattr_l__mod___blocks_2_blocks_0___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg179_1, reinterpret_tensor(buf345, (3208, 128), (128, 1), 0), reinterpret_tensor(arg178_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf346)
    del arg178_1
    del arg179_1
    # Source Nodes: [x_115], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf347 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf346, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf346, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf346, (8, 4, 401, 32), (153984, 32, 384, 1), 256))
    buf348 = buf347[0]
    del buf347
    buf355 = reinterpret_tensor(buf345, (3208, 128), (128, 1), 0); del buf345  # reuse
    # Source Nodes: [x_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg181_1, reinterpret_tensor(buf348, (3208, 128), (128, 1), 0), reinterpret_tensor(arg180_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf355)
    del arg180_1
    del arg181_1
    buf356 = buf343; del buf343  # reuse
    buf357 = buf342; del buf342  # reuse
    buf363 = buf295; del buf295  # reuse
    buf364 = buf294; del buf294  # reuse
    buf366 = buf293; del buf293  # reuse
    cpp_fused_add_native_layer_norm_52(c_void_p(buf341.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf366.data_ptr()))
    del arg188_1
    del arg189_1
    buf367 = reinterpret_tensor(buf285, (1576, 768), (768, 1), 0); del buf285  # reuse
    # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg191_1, reinterpret_tensor(buf366, (1576, 256), (256, 1), 0), reinterpret_tensor(arg190_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf367)
    del arg190_1
    del arg191_1
    # Source Nodes: [x_127], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf368 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf367, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf367, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf367, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf369 = buf368[0]
    del buf368
    buf376 = reinterpret_tensor(buf366, (1576, 256), (256, 1), 0); del buf366  # reuse
    # Source Nodes: [x_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg193_1, reinterpret_tensor(buf369, (1576, 256), (256, 1), 0), reinterpret_tensor(arg192_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf376)
    del arg192_1
    del arg193_1
    buf377 = buf364; del buf364  # reuse
    buf378 = buf363; del buf363  # reuse
    buf380 = reinterpret_tensor(buf369, (8, 197, 256), (50432, 256, 1), 0); del buf369  # reuse
    cpp_fused_add_native_layer_norm_53(c_void_p(buf362.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf380.data_ptr()))
    del arg194_1
    del arg195_1
    buf381 = buf367; del buf367  # reuse
    # Source Nodes: [x_132], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg197_1, reinterpret_tensor(buf380, (1576, 256), (256, 1), 0), reinterpret_tensor(arg196_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf381)
    del arg196_1
    del arg197_1
    buf382 = reinterpret_tensor(buf381, (8, 197, 768), (151296, 768, 1), 0); del buf381  # reuse
    cpp_fused_gelu_54(c_void_p(buf382.data_ptr()))
    buf383 = reinterpret_tensor(buf380, (1576, 256), (256, 1), 0); del buf380  # reuse
    # Source Nodes: [x_136], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg199_1, reinterpret_tensor(buf382, (1576, 768), (768, 1), 0), reinterpret_tensor(arg198_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf383)
    del arg198_1
    del arg199_1
    buf384 = buf378; del buf378  # reuse
    buf385 = buf377; del buf377  # reuse
    buf387 = reinterpret_tensor(buf308, (8, 197, 256), (50432, 256, 1), 0); del buf308  # reuse
    cpp_fused_add_native_layer_norm_55(c_void_p(buf362.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf387.data_ptr()))
    del arg200_1
    del arg201_1
    buf388 = reinterpret_tensor(buf382, (1576, 768), (768, 1), 0); del buf382  # reuse
    # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg203_1, reinterpret_tensor(buf387, (1576, 256), (256, 1), 0), reinterpret_tensor(arg202_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf388)
    del arg202_1
    del arg203_1
    # Source Nodes: [x_139], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf389 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf388, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf388, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf388, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf390 = buf389[0]
    del buf389
    buf397 = reinterpret_tensor(buf387, (1576, 256), (256, 1), 0); del buf387  # reuse
    # Source Nodes: [x_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg205_1, reinterpret_tensor(buf390, (1576, 256), (256, 1), 0), reinterpret_tensor(arg204_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf397)
    del arg204_1
    del arg205_1
    buf398 = buf385; del buf385  # reuse
    buf399 = buf384; del buf384  # reuse
    buf401 = reinterpret_tensor(buf390, (8, 197, 256), (50432, 256, 1), 0); del buf390  # reuse
    cpp_fused_add_native_layer_norm_56(c_void_p(buf362.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf401.data_ptr()))
    del arg206_1
    del arg207_1
    buf402 = buf388; del buf388  # reuse
    # Source Nodes: [x_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg209_1, reinterpret_tensor(buf401, (1576, 256), (256, 1), 0), reinterpret_tensor(arg208_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf402)
    del arg208_1
    del arg209_1
    buf403 = reinterpret_tensor(buf402, (8, 197, 768), (151296, 768, 1), 0); del buf402  # reuse
    cpp_fused_gelu_57(c_void_p(buf403.data_ptr()))
    buf404 = reinterpret_tensor(buf401, (1576, 256), (256, 1), 0); del buf401  # reuse
    # Source Nodes: [x_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg211_1, reinterpret_tensor(buf403, (1576, 768), (768, 1), 0), reinterpret_tensor(arg210_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf404)
    del arg210_1
    del arg211_1
    buf405 = reinterpret_tensor(buf404, (8, 197, 256), (50432, 256, 1), 0); del buf404  # reuse
    buf406 = buf399; del buf399  # reuse
    buf407 = buf398; del buf398  # reuse
    buf409 = reinterpret_tensor(buf306, (8, 197, 256), (50432, 256, 1), 0); del buf306  # reuse
    cpp_fused_add_native_layer_norm_58(c_void_p(buf405.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf409.data_ptr()))
    del arg212_1
    del arg213_1
    del buf362
    buf410 = reinterpret_tensor(buf403, (1576, 768), (768, 1), 0); del buf403  # reuse
    # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg215_1, reinterpret_tensor(buf409, (1576, 256), (256, 1), 0), reinterpret_tensor(arg214_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf410)
    del arg214_1
    del arg215_1
    # Source Nodes: [x_151], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf411 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf410, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf410, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf410, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf412 = buf411[0]
    del buf411
    buf419 = reinterpret_tensor(buf409, (1576, 256), (256, 1), 0); del buf409  # reuse
    # Source Nodes: [x_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg217_1, reinterpret_tensor(buf412, (1576, 256), (256, 1), 0), reinterpret_tensor(arg216_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf419)
    del arg216_1
    del arg217_1
    buf420 = buf407; del buf407  # reuse
    buf421 = buf406; del buf406  # reuse
    buf423 = reinterpret_tensor(buf348, (8, 401, 128), (51328, 128, 1), 0); del buf348  # reuse
    cpp_fused_add_native_layer_norm_59(c_void_p(buf405.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf423.data_ptr()))
    del arg182_1
    del arg183_1
    buf424 = buf346; del buf346  # reuse
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg185_1, reinterpret_tensor(buf423, (3208, 128), (128, 1), 0), reinterpret_tensor(arg184_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf424)
    del arg184_1
    del arg185_1
    buf425 = reinterpret_tensor(buf424, (8, 401, 384), (153984, 384, 1), 0); del buf424  # reuse
    cpp_fused_gelu_60(c_void_p(buf425.data_ptr()))
    buf426 = reinterpret_tensor(buf423, (3208, 128), (128, 1), 0); del buf423  # reuse
    # Source Nodes: [x_124], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg187_1, reinterpret_tensor(buf425, (3208, 384), (384, 1), 0), reinterpret_tensor(arg186_1, (384, 128), (1, 384), 0), alpha=1, beta=1, out=buf426)
    del arg186_1
    del arg187_1
    del buf425
    buf427 = buf312; del buf312  # reuse
    buf428 = buf311; del buf311  # reuse
    buf430 = reinterpret_tensor(buf412, (8, 197, 256), (50432, 256, 1), 0); del buf412  # reuse
    cpp_fused_add_native_layer_norm_61(c_void_p(buf341.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf430.data_ptr()))
    del arg218_1
    del arg219_1
    buf431 = buf410; del buf410  # reuse
    # Source Nodes: [x_156], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg221_1, reinterpret_tensor(buf430, (1576, 256), (256, 1), 0), reinterpret_tensor(arg220_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf431)
    del arg220_1
    del arg221_1
    buf432 = reinterpret_tensor(buf431, (8, 197, 768), (151296, 768, 1), 0); del buf431  # reuse
    cpp_fused_gelu_62(c_void_p(buf432.data_ptr()))
    buf433 = reinterpret_tensor(buf430, (1576, 256), (256, 1), 0); del buf430  # reuse
    # Source Nodes: [x_160], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg223_1, reinterpret_tensor(buf432, (1576, 768), (768, 1), 0), reinterpret_tensor(arg222_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf433)
    del arg222_1
    del arg223_1
    del buf432
    buf434 = buf191; del buf191  # reuse
    buf435 = buf190; del buf190  # reuse
    buf437 = reinterpret_tensor(buf340, (8, 1, 128), (128, 1024, 1), 0); del buf340  # reuse
    buf438 = reinterpret_tensor(buf437, (8, 1, 128), (128, 128, 1), 0); del buf437  # reuse
    cpp_fused_gelu_native_layer_norm_63(c_void_p(buf438.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()))
    del arg224_1
    del arg225_1
    buf439 = reinterpret_tensor(buf339, (8, 256), (256, 1), 0); del buf339  # reuse
    # Source Nodes: [l__mod___blocks_2_projs_0_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg227_1, reinterpret_tensor(buf438, (8, 128), (128, 1), 0), reinterpret_tensor(arg226_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf439)
    del arg226_1
    del arg227_1
    buf440 = buf421; del buf421  # reuse
    buf441 = buf420; del buf420  # reuse
    buf443 = reinterpret_tensor(buf397, (8, 197, 256), (50432, 256, 1), 0); del buf397  # reuse
    cpp_fused_cat_native_layer_norm_64(c_void_p(buf439.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf443.data_ptr()))
    del arg232_1
    del arg233_1
    buf444 = buf310; del buf310  # reuse
    # Source Nodes: [l__mod___blocks_2_fusion_0_attn_wq], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf443, (8, 256), (50432, 1), 0), reinterpret_tensor(arg234_1, (256, 256), (1, 256), 0), out=buf444)
    del arg234_1
    buf445 = buf383; del buf383  # reuse
    # Source Nodes: [l__mod___blocks_2_fusion_0_attn_wk], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg237_1, reinterpret_tensor(buf443, (1576, 256), (256, 1), 0), reinterpret_tensor(arg236_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf445)
    del arg236_1
    del arg237_1
    buf446 = reinterpret_tensor(buf444, (8, 1, 256), (256, 256, 1), 0); del buf444  # reuse
    buf447 = reinterpret_tensor(buf376, (8, 4, 64, 197), (50432, 12608, 197, 1), 0); del buf376  # reuse
    cpp_fused_add_clone_65(c_void_p(buf446.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf447.data_ptr()))
    del arg235_1
    del buf445
    buf448 = reinterpret_tensor(buf307, (32, 1, 197), (197, 197, 1), 0); del buf307  # reuse
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf446, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf447, (32, 64, 197), (12608, 197, 1), 0), out=buf448)
    buf449 = buf305; del buf305  # reuse
    buf450 = reinterpret_tensor(buf448, (8, 4, 1, 197), (788, 197, 6304, 1), 0); del buf448  # reuse
    buf451 = buf303; del buf303  # reuse
    cpp_fused__softmax_mul_66(c_void_p(buf450.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf451.data_ptr()))
    buf452 = reinterpret_tensor(buf447, (1576, 256), (256, 1), 0); del buf447  # reuse
    # Source Nodes: [l__mod___blocks_2_fusion_0_attn_wv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg239_1, reinterpret_tensor(buf443, (1576, 256), (256, 1), 0), reinterpret_tensor(arg238_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf452)
    del arg238_1
    del arg239_1
    buf453 = reinterpret_tensor(buf450, (8, 4, 1, 197), (788, 197, 197, 1), 0); del buf450  # reuse
    buf454 = reinterpret_tensor(buf443, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf443  # reuse
    cpp_fused__softmax_clone_67(c_void_p(buf453.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf454.data_ptr()))
    del buf452
    buf455 = reinterpret_tensor(buf446, (32, 1, 64), (64, 64, 1), 0); del buf446  # reuse
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf453, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf454, (32, 197, 64), (12608, 64, 1), 0), out=buf455)
    del buf453
    del buf454
    buf456 = buf166; del buf166  # reuse
    # Source Nodes: [x_164], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg241_1, reinterpret_tensor(buf455, (8, 256), (256, 1), 0), reinterpret_tensor(arg240_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf456)
    del arg240_1
    del arg241_1
    buf457 = reinterpret_tensor(buf456, (8, 1, 256), (256, 2048, 1), 0); del buf456  # reuse
    buf461 = reinterpret_tensor(buf455, (8, 1, 256), (256, 2048, 1), 0); del buf455  # reuse
    buf458 = buf428; del buf428  # reuse
    buf459 = buf427; del buf427  # reuse
    buf462 = reinterpret_tensor(buf461, (8, 1, 256), (256, 256, 1), 0); del buf461  # reuse
    cpp_fused_add_gelu_native_layer_norm_68(c_void_p(buf457.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()))
    del arg228_1
    del arg229_1
    del buf434
    del buf435
    del buf439
    buf463 = reinterpret_tensor(buf438, (8, 128), (128, 1), 0); del buf438  # reuse
    # Source Nodes: [l__mod___blocks_2_projs_1_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg231_1, reinterpret_tensor(buf462, (8, 256), (256, 1), 0), reinterpret_tensor(arg230_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf463)
    del arg230_1
    del arg231_1
    buf485 = reinterpret_tensor(buf462, (8, 1, 256), (256, 2048, 1), 0); del buf462  # reuse
    buf486 = reinterpret_tensor(buf485, (8, 1, 256), (256, 256, 1), 0); del buf485  # reuse
    cpp_fused_gelu_native_layer_norm_69(c_void_p(buf486.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()))
    del arg242_1
    del arg243_1
    buf487 = buf195; del buf195  # reuse
    # Source Nodes: [reverted_proj_cls_token_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg245_1, reinterpret_tensor(buf486, (8, 256), (256, 1), 0), reinterpret_tensor(arg244_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf487)
    del arg244_1
    del arg245_1
    buf464 = buf357; del buf357  # reuse
    buf465 = buf356; del buf356  # reuse
    buf488 = buf174; del buf174  # reuse
    buf489 = buf173; del buf173  # reuse
    buf467 = reinterpret_tensor(buf279, (8, 401, 128), (51328, 128, 1), 0); del buf279  # reuse
    cpp_fused_cat_native_layer_norm_70(c_void_p(buf463.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf467.data_ptr()))
    del arg246_1
    del arg247_1
    del buf464
    del buf465
    buf468 = buf334; del buf334  # reuse
    # Source Nodes: [l__mod___blocks_2_fusion_1_attn_wq], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf467, (8, 128), (51328, 1), 0), reinterpret_tensor(arg248_1, (128, 128), (1, 128), 0), out=buf468)
    del arg248_1
    buf469 = buf209; del buf209  # reuse
    # Source Nodes: [l__mod___blocks_2_fusion_1_attn_wk], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg251_1, reinterpret_tensor(buf467, (3208, 128), (128, 1), 0), reinterpret_tensor(arg250_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf469)
    del arg250_1
    del arg251_1
    buf470 = reinterpret_tensor(buf468, (8, 1, 128), (128, 128, 1), 0); del buf468  # reuse
    buf471 = reinterpret_tensor(buf135, (8, 4, 32, 401), (51328, 12832, 401, 1), 0); del buf135  # reuse
    cpp_fused_add_clone_71(c_void_p(buf470.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf471.data_ptr()))
    del arg249_1
    del buf469
    buf472 = reinterpret_tensor(buf331, (32, 1, 401), (401, 401, 1), 0); del buf331  # reuse
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf470, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf471, (32, 32, 401), (12832, 401, 1), 0), out=buf472)
    buf473 = buf451; del buf451  # reuse
    buf474 = reinterpret_tensor(buf472, (8, 4, 1, 401), (1604, 401, 12832, 1), 0); del buf472  # reuse
    buf475 = buf449; del buf449  # reuse
    cpp_fused__softmax_mul_72(c_void_p(buf474.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf475.data_ptr()))
    del buf473
    buf476 = reinterpret_tensor(buf471, (3208, 128), (128, 1), 0); del buf471  # reuse
    # Source Nodes: [l__mod___blocks_2_fusion_1_attn_wv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg253_1, reinterpret_tensor(buf467, (3208, 128), (128, 1), 0), reinterpret_tensor(arg252_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf476)
    del arg252_1
    del arg253_1
    buf477 = reinterpret_tensor(buf474, (8, 4, 1, 401), (1604, 401, 401, 1), 0); del buf474  # reuse
    buf478 = reinterpret_tensor(buf467, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf467  # reuse
    cpp_fused__softmax_clone_73(c_void_p(buf477.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf478.data_ptr()))
    del buf475
    del buf476
    buf479 = reinterpret_tensor(buf470, (32, 1, 32), (32, 32, 1), 0); del buf470  # reuse
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf477, (32, 1, 401), (401, 0, 1), 0), reinterpret_tensor(buf478, (32, 401, 32), (12832, 32, 1), 0), out=buf479)
    del buf477
    del buf478
    buf480 = buf189; del buf189  # reuse
    # Source Nodes: [x_168], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg255_1, reinterpret_tensor(buf479, (8, 128), (128, 1), 0), reinterpret_tensor(arg254_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf480)
    del arg254_1
    del arg255_1
    buf481 = reinterpret_tensor(buf480, (8, 1, 128), (128, 1024, 1), 0); del buf480  # reuse
    buf497 = reinterpret_tensor(buf479, (8, 128), (128, 1), 0); del buf479  # reuse
    buf482 = buf459; del buf459  # reuse
    buf483 = buf458; del buf458  # reuse
    buf491 = reinterpret_tensor(buf172, (8, 1, 128), (128, 1024, 1), 0); del buf172  # reuse
    buf492 = reinterpret_tensor(buf491, (8, 1, 128), (128, 128, 1), 0); del buf491  # reuse
    cpp_fused_add_clone_gelu_native_layer_norm_74(c_void_p(buf481.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()))
    del arg256_1
    del arg257_1
    del arg260_1
    del arg261_1
    del buf341
    del buf355
    del buf426
    del buf463
    del buf481
    del buf482
    del buf483
    del buf487
    del buf488
    del buf489
    buf493 = reinterpret_tensor(buf486, (8, 256), (256, 1), 0); del buf486  # reuse
    # Source Nodes: [reverted_proj_cls_token_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg259_1, reinterpret_tensor(buf492, (8, 128), (128, 1), 0), reinterpret_tensor(arg258_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf493)
    del arg258_1
    del arg259_1
    del buf492
    buf494 = buf441; del buf441  # reuse
    buf495 = buf440; del buf440  # reuse
    cpp_fused_cat_native_layer_norm_75(c_void_p(buf493.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()))
    buf498 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___head_0, l__mod___head_drop], Original ATen: [aten.addmm, aten.clone]
    extern_kernels.addmm(arg265_1, buf497, reinterpret_tensor(arg264_1, (128, 1000), (1, 128), 0), alpha=1, beta=1, out=buf498)
    del arg264_1
    del arg265_1
    del buf497
    buf499 = reinterpret_tensor(buf457, (8, 256), (256, 1), 0); del buf457  # reuse
    cpp_fused_clone_76(c_void_p(buf493.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(buf499.data_ptr()))
    del arg262_1
    del arg263_1
    del buf405
    del buf419
    del buf433
    del buf493
    del buf494
    del buf495
    buf500 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___head_1, l__mod___head_drop_1], Original ATen: [aten.addmm, aten.clone]
    extern_kernels.addmm(arg267_1, buf499, reinterpret_tensor(arg266_1, (256, 1000), (1, 256), 0), alpha=1, beta=1, out=buf500)
    del arg266_1
    del arg267_1
    del buf499
    buf501 = empty((8, 1000), device='cpu', dtype=torch.float32)
    cpp_fused_mean_77(c_void_p(buf498.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()))
    return (buf501, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, 401, 128), (51328, 128, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1, 256), (256, 256, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((1, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((128, 3, 12, 12), (432, 144, 12, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((256, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((1000, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((1000, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((8, 3, 240, 240), (172800, 57600, 240, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('crossvit_9_240', benchmark_compiled_module)
