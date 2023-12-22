
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (147L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (147L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(112);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_out_ptr0 + static_cast<long>((-7232L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp11(), to_float_mask(tmp10));
                            auto tmp14 = c10::convert<int>(2L*x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(in_out_ptr0 + static_cast<long>((-7168L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp18));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp19(), to_float_mask(tmp18));
                            auto tmp22 = at::vec::maximum(tmp21, tmp13);
                            auto tmp23 = c10::convert<int>(1L + (2L*x2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = masked_load(in_out_ptr0 + static_cast<long>((-7104L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp27));
                                return tmp29;
                            }
                            ;
                            auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp28(), to_float_mask(tmp27));
                            auto tmp31 = at::vec::maximum(tmp30, tmp22);
                            auto tmp32 = c10::convert<int>(2L*x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = masked_load(in_out_ptr0 + static_cast<long>((-64L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = at::vec::maximum(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(in_out_ptr0 + static_cast<long>(x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = at::vec::maximum(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(in_out_ptr0 + static_cast<long>(64L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp46));
                                return tmp48;
                            }
                            ;
                            auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp47(), to_float_mask(tmp46));
                            auto tmp50 = at::vec::maximum(tmp49, tmp45);
                            auto tmp51 = c10::convert<int>(1L + (2L*x1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = masked_load(in_out_ptr0 + static_cast<long>(7104L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = at::vec::maximum(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(in_out_ptr0 + static_cast<long>(7168L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = at::vec::maximum(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(in_out_ptr0 + static_cast<long>(7232L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = at::vec::maximum(tmp68, tmp64);
                            tmp69.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (3584L*x1) + (200704L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x2 + (9L*x1) + (36L*x0))];
                            out_ptr0[static_cast<long>(x1 + (4L*x2) + (36L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (9L*x1) + (36L*x0))];
                    out_ptr0[static_cast<long>(x1 + (4L*x2) + (36L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (9L*x1) + (36L*x0))];
                    out_ptr0[static_cast<long>(x1 + (4L*x2) + (36L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_5 = async_compile.cpp('''
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
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-1L) + x2);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(56);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>((-1L) + x3);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr0[static_cast<long>((-7200L) + x1 + (128L*x3) + (7168L*x2) + (401408L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = c10::convert<long>(x3);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr0[static_cast<long>((-7072L) + x1 + (128L*x3) + (7168L*x2) + (401408L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = decltype(tmp21)(tmp21 + tmp13);
                            auto tmp23 = c10::convert<long>(1L + x3);
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr0[static_cast<long>((-6944L) + x1 + (128L*x3) + (7168L*x2) + (401408L*x0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp31 = decltype(tmp30)(tmp30 + tmp22);
                            auto tmp32 = c10::convert<long>(x2);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = in_ptr0[static_cast<long>((-32L) + x1 + (128L*x3) + (7168L*x2) + (401408L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                            auto tmp40 = decltype(tmp39)(tmp39 + tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = in_ptr0[static_cast<long>(96L + x1 + (128L*x3) + (7168L*x2) + (401408L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : static_cast<decltype(tmp42())>(0.0);
                            auto tmp45 = decltype(tmp44)(tmp44 + tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = in_ptr0[static_cast<long>(224L + x1 + (128L*x3) + (7168L*x2) + (401408L*x0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : static_cast<decltype(tmp47())>(0.0);
                            auto tmp50 = decltype(tmp49)(tmp49 + tmp45);
                            auto tmp51 = c10::convert<long>(1L + x2);
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = in_ptr0[static_cast<long>(7136L + x1 + (128L*x3) + (7168L*x2) + (401408L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                            auto tmp59 = decltype(tmp58)(tmp58 + tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = in_ptr0[static_cast<long>(7264L + x1 + (128L*x3) + (7168L*x2) + (401408L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : static_cast<decltype(tmp61())>(0.0);
                            auto tmp64 = decltype(tmp63)(tmp63 + tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = in_ptr0[static_cast<long>(7392L + x1 + (128L*x3) + (7168L*x2) + (401408L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : static_cast<decltype(tmp66())>(0.0);
                            auto tmp69 = decltype(tmp68)(tmp68 + tmp64);
                            auto tmp70 = static_cast<long>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<long>(57);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<long>((-1L) + x2);
                                auto tmp81 = static_cast<long>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<long>(56);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<long>((-1L) + x3);
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp91 = [&]
                                {
                                    auto tmp92 = static_cast<float>(1.0);
                                    return tmp92;
                                }
                                ;
                                auto tmp93 = tmp90 ? tmp91() : static_cast<decltype(tmp91())>(1.0);
                                return tmp93;
                            }
                            ;
                            auto tmp94 = tmp78 ? tmp79() : static_cast<decltype(tmp79())>(0.0);
                            auto tmp95 = tmp14 >= tmp70;
                            auto tmp96 = tmp14 < tmp72;
                            auto tmp97 = tmp95 & tmp96;
                            auto tmp98 = tmp74 & tmp97;
                            auto tmp99 = [&]
                            {
                                auto tmp100 = c10::convert<long>((-1L) + x2);
                                auto tmp101 = static_cast<long>(0);
                                auto tmp102 = tmp100 >= tmp101;
                                auto tmp103 = static_cast<long>(56);
                                auto tmp104 = tmp100 < tmp103;
                                auto tmp105 = tmp102 & tmp104;
                                auto tmp106 = c10::convert<long>(x3);
                                auto tmp107 = tmp106 >= tmp101;
                                auto tmp108 = tmp106 < tmp103;
                                auto tmp109 = tmp107 & tmp108;
                                auto tmp110 = tmp105 & tmp109;
                                auto tmp111 = [&]
                                {
                                    auto tmp112 = static_cast<float>(1.0);
                                    return tmp112;
                                }
                                ;
                                auto tmp113 = tmp110 ? tmp111() : static_cast<decltype(tmp111())>(1.0);
                                return tmp113;
                            }
                            ;
                            auto tmp114 = tmp98 ? tmp99() : static_cast<decltype(tmp99())>(0.0);
                            auto tmp115 = decltype(tmp114)(tmp114 + tmp94);
                            auto tmp116 = tmp23 >= tmp70;
                            auto tmp117 = tmp23 < tmp72;
                            auto tmp118 = tmp116 & tmp117;
                            auto tmp119 = tmp74 & tmp118;
                            auto tmp120 = [&]
                            {
                                auto tmp121 = c10::convert<long>((-1L) + x2);
                                auto tmp122 = static_cast<long>(0);
                                auto tmp123 = tmp121 >= tmp122;
                                auto tmp124 = static_cast<long>(56);
                                auto tmp125 = tmp121 < tmp124;
                                auto tmp126 = tmp123 & tmp125;
                                auto tmp127 = c10::convert<long>(1L + x3);
                                auto tmp128 = tmp127 >= tmp122;
                                auto tmp129 = tmp127 < tmp124;
                                auto tmp130 = tmp128 & tmp129;
                                auto tmp131 = tmp126 & tmp130;
                                auto tmp132 = [&]
                                {
                                    auto tmp133 = static_cast<float>(1.0);
                                    return tmp133;
                                }
                                ;
                                auto tmp134 = tmp131 ? tmp132() : static_cast<decltype(tmp132())>(1.0);
                                return tmp134;
                            }
                            ;
                            auto tmp135 = tmp119 ? tmp120() : static_cast<decltype(tmp120())>(0.0);
                            auto tmp136 = decltype(tmp135)(tmp135 + tmp115);
                            auto tmp137 = tmp32 >= tmp70;
                            auto tmp138 = tmp32 < tmp72;
                            auto tmp139 = tmp137 & tmp138;
                            auto tmp140 = tmp139 & tmp77;
                            auto tmp141 = [&]
                            {
                                auto tmp142 = c10::convert<long>(x2);
                                auto tmp143 = static_cast<long>(0);
                                auto tmp144 = tmp142 >= tmp143;
                                auto tmp145 = static_cast<long>(56);
                                auto tmp146 = tmp142 < tmp145;
                                auto tmp147 = tmp144 & tmp146;
                                auto tmp148 = c10::convert<long>((-1L) + x3);
                                auto tmp149 = tmp148 >= tmp143;
                                auto tmp150 = tmp148 < tmp145;
                                auto tmp151 = tmp149 & tmp150;
                                auto tmp152 = tmp147 & tmp151;
                                auto tmp153 = [&]
                                {
                                    auto tmp154 = static_cast<float>(1.0);
                                    return tmp154;
                                }
                                ;
                                auto tmp155 = tmp152 ? tmp153() : static_cast<decltype(tmp153())>(1.0);
                                return tmp155;
                            }
                            ;
                            auto tmp156 = tmp140 ? tmp141() : static_cast<decltype(tmp141())>(0.0);
                            auto tmp157 = decltype(tmp156)(tmp156 + tmp136);
                            auto tmp158 = tmp139 & tmp97;
                            auto tmp159 = [&]
                            {
                                auto tmp160 = c10::convert<long>(x2);
                                auto tmp161 = static_cast<long>(0);
                                auto tmp162 = tmp160 >= tmp161;
                                auto tmp163 = static_cast<long>(56);
                                auto tmp164 = tmp160 < tmp163;
                                auto tmp165 = tmp162 & tmp164;
                                auto tmp166 = c10::convert<long>(x3);
                                auto tmp167 = tmp166 >= tmp161;
                                auto tmp168 = tmp166 < tmp163;
                                auto tmp169 = tmp167 & tmp168;
                                auto tmp170 = tmp165 & tmp169;
                                auto tmp171 = [&]
                                {
                                    auto tmp172 = static_cast<float>(1.0);
                                    return tmp172;
                                }
                                ;
                                auto tmp173 = tmp170 ? tmp171() : static_cast<decltype(tmp171())>(1.0);
                                return tmp173;
                            }
                            ;
                            auto tmp174 = tmp158 ? tmp159() : static_cast<decltype(tmp159())>(0.0);
                            auto tmp175 = decltype(tmp174)(tmp174 + tmp157);
                            auto tmp176 = tmp139 & tmp118;
                            auto tmp177 = [&]
                            {
                                auto tmp178 = c10::convert<long>(x2);
                                auto tmp179 = static_cast<long>(0);
                                auto tmp180 = tmp178 >= tmp179;
                                auto tmp181 = static_cast<long>(56);
                                auto tmp182 = tmp178 < tmp181;
                                auto tmp183 = tmp180 & tmp182;
                                auto tmp184 = c10::convert<long>(1L + x3);
                                auto tmp185 = tmp184 >= tmp179;
                                auto tmp186 = tmp184 < tmp181;
                                auto tmp187 = tmp185 & tmp186;
                                auto tmp188 = tmp183 & tmp187;
                                auto tmp189 = [&]
                                {
                                    auto tmp190 = static_cast<float>(1.0);
                                    return tmp190;
                                }
                                ;
                                auto tmp191 = tmp188 ? tmp189() : static_cast<decltype(tmp189())>(1.0);
                                return tmp191;
                            }
                            ;
                            auto tmp192 = tmp176 ? tmp177() : static_cast<decltype(tmp177())>(0.0);
                            auto tmp193 = decltype(tmp192)(tmp192 + tmp175);
                            auto tmp194 = tmp51 >= tmp70;
                            auto tmp195 = tmp51 < tmp72;
                            auto tmp196 = tmp194 & tmp195;
                            auto tmp197 = tmp196 & tmp77;
                            auto tmp198 = [&]
                            {
                                auto tmp199 = c10::convert<long>(1L + x2);
                                auto tmp200 = static_cast<long>(0);
                                auto tmp201 = tmp199 >= tmp200;
                                auto tmp202 = static_cast<long>(56);
                                auto tmp203 = tmp199 < tmp202;
                                auto tmp204 = tmp201 & tmp203;
                                auto tmp205 = c10::convert<long>((-1L) + x3);
                                auto tmp206 = tmp205 >= tmp200;
                                auto tmp207 = tmp205 < tmp202;
                                auto tmp208 = tmp206 & tmp207;
                                auto tmp209 = tmp204 & tmp208;
                                auto tmp210 = [&]
                                {
                                    auto tmp211 = static_cast<float>(1.0);
                                    return tmp211;
                                }
                                ;
                                auto tmp212 = tmp209 ? tmp210() : static_cast<decltype(tmp210())>(1.0);
                                return tmp212;
                            }
                            ;
                            auto tmp213 = tmp197 ? tmp198() : static_cast<decltype(tmp198())>(0.0);
                            auto tmp214 = decltype(tmp213)(tmp213 + tmp193);
                            auto tmp215 = tmp196 & tmp97;
                            auto tmp216 = [&]
                            {
                                auto tmp217 = c10::convert<long>(1L + x2);
                                auto tmp218 = static_cast<long>(0);
                                auto tmp219 = tmp217 >= tmp218;
                                auto tmp220 = static_cast<long>(56);
                                auto tmp221 = tmp217 < tmp220;
                                auto tmp222 = tmp219 & tmp221;
                                auto tmp223 = c10::convert<long>(x3);
                                auto tmp224 = tmp223 >= tmp218;
                                auto tmp225 = tmp223 < tmp220;
                                auto tmp226 = tmp224 & tmp225;
                                auto tmp227 = tmp222 & tmp226;
                                auto tmp228 = [&]
                                {
                                    auto tmp229 = static_cast<float>(1.0);
                                    return tmp229;
                                }
                                ;
                                auto tmp230 = tmp227 ? tmp228() : static_cast<decltype(tmp228())>(1.0);
                                return tmp230;
                            }
                            ;
                            auto tmp231 = tmp215 ? tmp216() : static_cast<decltype(tmp216())>(0.0);
                            auto tmp232 = decltype(tmp231)(tmp231 + tmp214);
                            auto tmp233 = tmp196 & tmp118;
                            auto tmp234 = [&]
                            {
                                auto tmp235 = c10::convert<long>(1L + x2);
                                auto tmp236 = static_cast<long>(0);
                                auto tmp237 = tmp235 >= tmp236;
                                auto tmp238 = static_cast<long>(56);
                                auto tmp239 = tmp235 < tmp238;
                                auto tmp240 = tmp237 & tmp239;
                                auto tmp241 = c10::convert<long>(1L + x3);
                                auto tmp242 = tmp241 >= tmp236;
                                auto tmp243 = tmp241 < tmp238;
                                auto tmp244 = tmp242 & tmp243;
                                auto tmp245 = tmp240 & tmp244;
                                auto tmp246 = [&]
                                {
                                    auto tmp247 = static_cast<float>(1.0);
                                    return tmp247;
                                }
                                ;
                                auto tmp248 = tmp245 ? tmp246() : static_cast<decltype(tmp246())>(1.0);
                                return tmp248;
                            }
                            ;
                            auto tmp249 = tmp233 ? tmp234() : static_cast<decltype(tmp234())>(0.0);
                            auto tmp250 = decltype(tmp249)(tmp249 + tmp232);
                            auto tmp251 = tmp69 / tmp250;
                            out_ptr0[static_cast<long>(x3 + (56L*x2) + (3136L*x1) + (401408L*x0))] = tmp251;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (32L*x2) + (100352L*x0)), static_cast<long>(32L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr1 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x1 + (32L*x2) + (100352L*x0)), static_cast<long>(32L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr10[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr2 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr11 + static_cast<long>(x1 + (32L*x2) + (100352L*x0)), static_cast<long>(32L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr12[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr13[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr14[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr15[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr3 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (128L*x2) + (401408L*x0)), static_cast<long>(128L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.cpp('''
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
                       const float* in_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = tmp16 + tmp29;
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x2 + (9L*x1) + (36L*x0))];
                            out_ptr0[static_cast<long>(x1 + (4L*x2) + (36L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_8 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (100352L*x0)), static_cast<long>(32L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (3136L*x2) + (401408L*x0)), static_cast<long>(3136L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x2 + (128L*x1) + (128L*x1_inner) + (401408L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (32L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr6[static_cast<long>(x2 + (9L*x1) + (36L*x0))];
                            out_ptr2[static_cast<long>(x1 + (4L*x2) + (36L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_9 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (100352L*x0)), static_cast<long>(32L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (3136L*x2) + (401408L*x0)), static_cast<long>(3136L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(64L + x2 + (128L*x1) + (128L*x1_inner) + (401408L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (32L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr6[static_cast<long>(x2 + (9L*x1) + (36L*x0))];
                            out_ptr2[static_cast<long>(x1 + (4L*x2) + (36L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_10 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (100352L*x0)), static_cast<long>(32L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(96L + x1 + (128L*x2) + (401408L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (128L*x2) + (401408L*x0)), static_cast<long>(128L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x2 + (9L*x1) + (36L*x0))];
                            out_ptr0[static_cast<long>(x1 + (4L*x2) + (36L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_13 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (100352L*x0)), static_cast<long>(32L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (3136L*x2) + (401408L*x0)), static_cast<long>(3136L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x2 + (128L*x1) + (128L*x1_inner) + (401408L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (32L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr6[static_cast<long>(x2 + (9L*x1) + (36L*x0))];
                            out_ptr2[static_cast<long>(x1 + (4L*x2) + (36L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_14 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (100352L*x0)), static_cast<long>(32L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (3136L*x2) + (401408L*x0)), static_cast<long>(3136L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(64L + x2 + (128L*x1) + (128L*x1_inner) + (401408L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (32L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr6[static_cast<long>(x2 + (9L*x1) + (36L*x0))];
                            out_ptr2[static_cast<long>(x1 + (4L*x2) + (36L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_15 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (100352L*x0)), static_cast<long>(32L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(96L + x1 + (128L*x2) + (401408L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (128L*x2) + (401408L*x0)), static_cast<long>(128L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_20 = async_compile.cpp('''
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
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-1L) + (2L*x2));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(56);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>((-1L) + (2L*x3));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr0[static_cast<long>((-14400L) + x1 + (512L*x3) + (28672L*x2) + (802816L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = c10::convert<long>(2L*x3);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr0[static_cast<long>((-14144L) + x1 + (512L*x3) + (28672L*x2) + (802816L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = decltype(tmp21)(tmp21 + tmp13);
                            auto tmp23 = c10::convert<long>(1L + (2L*x3));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr0[static_cast<long>((-13888L) + x1 + (512L*x3) + (28672L*x2) + (802816L*x0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp31 = decltype(tmp30)(tmp30 + tmp22);
                            auto tmp32 = c10::convert<long>(2L*x2);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = in_ptr0[static_cast<long>((-64L) + x1 + (512L*x3) + (28672L*x2) + (802816L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                            auto tmp40 = decltype(tmp39)(tmp39 + tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = in_ptr0[static_cast<long>(192L + x1 + (512L*x3) + (28672L*x2) + (802816L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : static_cast<decltype(tmp42())>(0.0);
                            auto tmp45 = decltype(tmp44)(tmp44 + tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = in_ptr0[static_cast<long>(448L + x1 + (512L*x3) + (28672L*x2) + (802816L*x0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : static_cast<decltype(tmp47())>(0.0);
                            auto tmp50 = decltype(tmp49)(tmp49 + tmp45);
                            auto tmp51 = c10::convert<long>(1L + (2L*x2));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = in_ptr0[static_cast<long>(14272L + x1 + (512L*x3) + (28672L*x2) + (802816L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                            auto tmp59 = decltype(tmp58)(tmp58 + tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = in_ptr0[static_cast<long>(14528L + x1 + (512L*x3) + (28672L*x2) + (802816L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : static_cast<decltype(tmp61())>(0.0);
                            auto tmp64 = decltype(tmp63)(tmp63 + tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = in_ptr0[static_cast<long>(14784L + x1 + (512L*x3) + (28672L*x2) + (802816L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : static_cast<decltype(tmp66())>(0.0);
                            auto tmp69 = decltype(tmp68)(tmp68 + tmp64);
                            auto tmp70 = static_cast<long>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<long>(57);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<long>((-1L) + (2L*x2));
                                auto tmp81 = static_cast<long>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<long>(56);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<long>((-1L) + (2L*x3));
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp91 = [&]
                                {
                                    auto tmp92 = static_cast<float>(1.0);
                                    return tmp92;
                                }
                                ;
                                auto tmp93 = tmp90 ? tmp91() : static_cast<decltype(tmp91())>(1.0);
                                return tmp93;
                            }
                            ;
                            auto tmp94 = tmp78 ? tmp79() : static_cast<decltype(tmp79())>(0.0);
                            auto tmp95 = tmp14 >= tmp70;
                            auto tmp96 = tmp14 < tmp72;
                            auto tmp97 = tmp95 & tmp96;
                            auto tmp98 = tmp74 & tmp97;
                            auto tmp99 = [&]
                            {
                                auto tmp100 = c10::convert<long>((-1L) + (2L*x2));
                                auto tmp101 = static_cast<long>(0);
                                auto tmp102 = tmp100 >= tmp101;
                                auto tmp103 = static_cast<long>(56);
                                auto tmp104 = tmp100 < tmp103;
                                auto tmp105 = tmp102 & tmp104;
                                auto tmp106 = c10::convert<long>(2L*x3);
                                auto tmp107 = tmp106 >= tmp101;
                                auto tmp108 = tmp106 < tmp103;
                                auto tmp109 = tmp107 & tmp108;
                                auto tmp110 = tmp105 & tmp109;
                                auto tmp111 = [&]
                                {
                                    auto tmp112 = static_cast<float>(1.0);
                                    return tmp112;
                                }
                                ;
                                auto tmp113 = tmp110 ? tmp111() : static_cast<decltype(tmp111())>(1.0);
                                return tmp113;
                            }
                            ;
                            auto tmp114 = tmp98 ? tmp99() : static_cast<decltype(tmp99())>(0.0);
                            auto tmp115 = decltype(tmp114)(tmp114 + tmp94);
                            auto tmp116 = tmp23 >= tmp70;
                            auto tmp117 = tmp23 < tmp72;
                            auto tmp118 = tmp116 & tmp117;
                            auto tmp119 = tmp74 & tmp118;
                            auto tmp120 = [&]
                            {
                                auto tmp121 = c10::convert<long>((-1L) + (2L*x2));
                                auto tmp122 = static_cast<long>(0);
                                auto tmp123 = tmp121 >= tmp122;
                                auto tmp124 = static_cast<long>(56);
                                auto tmp125 = tmp121 < tmp124;
                                auto tmp126 = tmp123 & tmp125;
                                auto tmp127 = c10::convert<long>(1L + (2L*x3));
                                auto tmp128 = tmp127 >= tmp122;
                                auto tmp129 = tmp127 < tmp124;
                                auto tmp130 = tmp128 & tmp129;
                                auto tmp131 = tmp126 & tmp130;
                                auto tmp132 = [&]
                                {
                                    auto tmp133 = static_cast<float>(1.0);
                                    return tmp133;
                                }
                                ;
                                auto tmp134 = tmp131 ? tmp132() : static_cast<decltype(tmp132())>(1.0);
                                return tmp134;
                            }
                            ;
                            auto tmp135 = tmp119 ? tmp120() : static_cast<decltype(tmp120())>(0.0);
                            auto tmp136 = decltype(tmp135)(tmp135 + tmp115);
                            auto tmp137 = tmp32 >= tmp70;
                            auto tmp138 = tmp32 < tmp72;
                            auto tmp139 = tmp137 & tmp138;
                            auto tmp140 = tmp139 & tmp77;
                            auto tmp141 = [&]
                            {
                                auto tmp142 = c10::convert<long>(2L*x2);
                                auto tmp143 = static_cast<long>(0);
                                auto tmp144 = tmp142 >= tmp143;
                                auto tmp145 = static_cast<long>(56);
                                auto tmp146 = tmp142 < tmp145;
                                auto tmp147 = tmp144 & tmp146;
                                auto tmp148 = c10::convert<long>((-1L) + (2L*x3));
                                auto tmp149 = tmp148 >= tmp143;
                                auto tmp150 = tmp148 < tmp145;
                                auto tmp151 = tmp149 & tmp150;
                                auto tmp152 = tmp147 & tmp151;
                                auto tmp153 = [&]
                                {
                                    auto tmp154 = static_cast<float>(1.0);
                                    return tmp154;
                                }
                                ;
                                auto tmp155 = tmp152 ? tmp153() : static_cast<decltype(tmp153())>(1.0);
                                return tmp155;
                            }
                            ;
                            auto tmp156 = tmp140 ? tmp141() : static_cast<decltype(tmp141())>(0.0);
                            auto tmp157 = decltype(tmp156)(tmp156 + tmp136);
                            auto tmp158 = tmp139 & tmp97;
                            auto tmp159 = [&]
                            {
                                auto tmp160 = c10::convert<long>(2L*x2);
                                auto tmp161 = static_cast<long>(0);
                                auto tmp162 = tmp160 >= tmp161;
                                auto tmp163 = static_cast<long>(56);
                                auto tmp164 = tmp160 < tmp163;
                                auto tmp165 = tmp162 & tmp164;
                                auto tmp166 = c10::convert<long>(2L*x3);
                                auto tmp167 = tmp166 >= tmp161;
                                auto tmp168 = tmp166 < tmp163;
                                auto tmp169 = tmp167 & tmp168;
                                auto tmp170 = tmp165 & tmp169;
                                auto tmp171 = [&]
                                {
                                    auto tmp172 = static_cast<float>(1.0);
                                    return tmp172;
                                }
                                ;
                                auto tmp173 = tmp170 ? tmp171() : static_cast<decltype(tmp171())>(1.0);
                                return tmp173;
                            }
                            ;
                            auto tmp174 = tmp158 ? tmp159() : static_cast<decltype(tmp159())>(0.0);
                            auto tmp175 = decltype(tmp174)(tmp174 + tmp157);
                            auto tmp176 = tmp139 & tmp118;
                            auto tmp177 = [&]
                            {
                                auto tmp178 = c10::convert<long>(2L*x2);
                                auto tmp179 = static_cast<long>(0);
                                auto tmp180 = tmp178 >= tmp179;
                                auto tmp181 = static_cast<long>(56);
                                auto tmp182 = tmp178 < tmp181;
                                auto tmp183 = tmp180 & tmp182;
                                auto tmp184 = c10::convert<long>(1L + (2L*x3));
                                auto tmp185 = tmp184 >= tmp179;
                                auto tmp186 = tmp184 < tmp181;
                                auto tmp187 = tmp185 & tmp186;
                                auto tmp188 = tmp183 & tmp187;
                                auto tmp189 = [&]
                                {
                                    auto tmp190 = static_cast<float>(1.0);
                                    return tmp190;
                                }
                                ;
                                auto tmp191 = tmp188 ? tmp189() : static_cast<decltype(tmp189())>(1.0);
                                return tmp191;
                            }
                            ;
                            auto tmp192 = tmp176 ? tmp177() : static_cast<decltype(tmp177())>(0.0);
                            auto tmp193 = decltype(tmp192)(tmp192 + tmp175);
                            auto tmp194 = tmp51 >= tmp70;
                            auto tmp195 = tmp51 < tmp72;
                            auto tmp196 = tmp194 & tmp195;
                            auto tmp197 = tmp196 & tmp77;
                            auto tmp198 = [&]
                            {
                                auto tmp199 = c10::convert<long>(1L + (2L*x2));
                                auto tmp200 = static_cast<long>(0);
                                auto tmp201 = tmp199 >= tmp200;
                                auto tmp202 = static_cast<long>(56);
                                auto tmp203 = tmp199 < tmp202;
                                auto tmp204 = tmp201 & tmp203;
                                auto tmp205 = c10::convert<long>((-1L) + (2L*x3));
                                auto tmp206 = tmp205 >= tmp200;
                                auto tmp207 = tmp205 < tmp202;
                                auto tmp208 = tmp206 & tmp207;
                                auto tmp209 = tmp204 & tmp208;
                                auto tmp210 = [&]
                                {
                                    auto tmp211 = static_cast<float>(1.0);
                                    return tmp211;
                                }
                                ;
                                auto tmp212 = tmp209 ? tmp210() : static_cast<decltype(tmp210())>(1.0);
                                return tmp212;
                            }
                            ;
                            auto tmp213 = tmp197 ? tmp198() : static_cast<decltype(tmp198())>(0.0);
                            auto tmp214 = decltype(tmp213)(tmp213 + tmp193);
                            auto tmp215 = tmp196 & tmp97;
                            auto tmp216 = [&]
                            {
                                auto tmp217 = c10::convert<long>(1L + (2L*x2));
                                auto tmp218 = static_cast<long>(0);
                                auto tmp219 = tmp217 >= tmp218;
                                auto tmp220 = static_cast<long>(56);
                                auto tmp221 = tmp217 < tmp220;
                                auto tmp222 = tmp219 & tmp221;
                                auto tmp223 = c10::convert<long>(2L*x3);
                                auto tmp224 = tmp223 >= tmp218;
                                auto tmp225 = tmp223 < tmp220;
                                auto tmp226 = tmp224 & tmp225;
                                auto tmp227 = tmp222 & tmp226;
                                auto tmp228 = [&]
                                {
                                    auto tmp229 = static_cast<float>(1.0);
                                    return tmp229;
                                }
                                ;
                                auto tmp230 = tmp227 ? tmp228() : static_cast<decltype(tmp228())>(1.0);
                                return tmp230;
                            }
                            ;
                            auto tmp231 = tmp215 ? tmp216() : static_cast<decltype(tmp216())>(0.0);
                            auto tmp232 = decltype(tmp231)(tmp231 + tmp214);
                            auto tmp233 = tmp196 & tmp118;
                            auto tmp234 = [&]
                            {
                                auto tmp235 = c10::convert<long>(1L + (2L*x2));
                                auto tmp236 = static_cast<long>(0);
                                auto tmp237 = tmp235 >= tmp236;
                                auto tmp238 = static_cast<long>(56);
                                auto tmp239 = tmp235 < tmp238;
                                auto tmp240 = tmp237 & tmp239;
                                auto tmp241 = c10::convert<long>(1L + (2L*x3));
                                auto tmp242 = tmp241 >= tmp236;
                                auto tmp243 = tmp241 < tmp238;
                                auto tmp244 = tmp242 & tmp243;
                                auto tmp245 = tmp240 & tmp244;
                                auto tmp246 = [&]
                                {
                                    auto tmp247 = static_cast<float>(1.0);
                                    return tmp247;
                                }
                                ;
                                auto tmp248 = tmp245 ? tmp246() : static_cast<decltype(tmp246())>(1.0);
                                return tmp248;
                            }
                            ;
                            auto tmp249 = tmp233 ? tmp234() : static_cast<decltype(tmp234())>(0.0);
                            auto tmp250 = decltype(tmp249)(tmp249 + tmp232);
                            auto tmp251 = tmp69 / tmp250;
                            out_ptr0[static_cast<long>(x3 + (28L*x2) + (784L*x1) + (200704L*x0))] = tmp251;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (50176L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr1 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x1 + (64L*x2) + (50176L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr10[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr2 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr11 + static_cast<long>(x1 + (64L*x2) + (50176L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr12[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr13[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr14[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr15[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr3 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (256L*x2) + (200704L*x0)), static_cast<long>(256L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_21 = async_compile.cpp('''
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
                       const float* in_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = tmp16 + tmp29;
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_23 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (50176L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (784L*x2) + (200704L*x0)), static_cast<long>(784L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(64L + x2 + (256L*x1) + (256L*x1_inner) + (200704L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (50176L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_24 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (50176L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (784L*x2) + (200704L*x0)), static_cast<long>(784L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(128L + x2 + (256L*x1) + (256L*x1_inner) + (200704L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (50176L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_25 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (50176L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(192L + x1 + (256L*x2) + (200704L*x0)), static_cast<long>(256L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (256L*x2) + (200704L*x0)), static_cast<long>(256L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_28 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (50176L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (784L*x2) + (200704L*x0)), static_cast<long>(784L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(64L + x2 + (256L*x1) + (256L*x1_inner) + (200704L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (50176L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_29 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (50176L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (784L*x2) + (200704L*x0)), static_cast<long>(784L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(128L + x2 + (256L*x1) + (256L*x1_inner) + (200704L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (50176L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_30 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (50176L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(192L + x1 + (256L*x2) + (200704L*x0)), static_cast<long>(256L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (256L*x2) + (200704L*x0)), static_cast<long>(256L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_33 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (50176L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (784L*x2) + (200704L*x0)), static_cast<long>(784L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(64L + x2 + (256L*x1) + (256L*x1_inner) + (200704L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (50176L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_34 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (50176L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (784L*x2) + (200704L*x0)), static_cast<long>(784L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(128L + x2 + (256L*x1) + (256L*x1_inner) + (200704L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (50176L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_35 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (50176L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(192L + x1 + (256L*x2) + (200704L*x0)), static_cast<long>(256L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (256L*x2) + (200704L*x0)), static_cast<long>(256L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_40 = async_compile.cpp('''
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
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-1L) + (2L*x2));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(28);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>((-1L) + (2L*x3));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr0[static_cast<long>((-14464L) + x1 + (1024L*x3) + (28672L*x2) + (401408L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = c10::convert<long>(2L*x3);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr0[static_cast<long>((-13952L) + x1 + (1024L*x3) + (28672L*x2) + (401408L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = decltype(tmp21)(tmp21 + tmp13);
                            auto tmp23 = c10::convert<long>(1L + (2L*x3));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr0[static_cast<long>((-13440L) + x1 + (1024L*x3) + (28672L*x2) + (401408L*x0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp31 = decltype(tmp30)(tmp30 + tmp22);
                            auto tmp32 = c10::convert<long>(2L*x2);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = in_ptr0[static_cast<long>((-128L) + x1 + (1024L*x3) + (28672L*x2) + (401408L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                            auto tmp40 = decltype(tmp39)(tmp39 + tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = in_ptr0[static_cast<long>(384L + x1 + (1024L*x3) + (28672L*x2) + (401408L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : static_cast<decltype(tmp42())>(0.0);
                            auto tmp45 = decltype(tmp44)(tmp44 + tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = in_ptr0[static_cast<long>(896L + x1 + (1024L*x3) + (28672L*x2) + (401408L*x0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : static_cast<decltype(tmp47())>(0.0);
                            auto tmp50 = decltype(tmp49)(tmp49 + tmp45);
                            auto tmp51 = c10::convert<long>(1L + (2L*x2));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = in_ptr0[static_cast<long>(14208L + x1 + (1024L*x3) + (28672L*x2) + (401408L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                            auto tmp59 = decltype(tmp58)(tmp58 + tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = in_ptr0[static_cast<long>(14720L + x1 + (1024L*x3) + (28672L*x2) + (401408L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : static_cast<decltype(tmp61())>(0.0);
                            auto tmp64 = decltype(tmp63)(tmp63 + tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = in_ptr0[static_cast<long>(15232L + x1 + (1024L*x3) + (28672L*x2) + (401408L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : static_cast<decltype(tmp66())>(0.0);
                            auto tmp69 = decltype(tmp68)(tmp68 + tmp64);
                            auto tmp70 = static_cast<long>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<long>(29);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<long>((-1L) + (2L*x2));
                                auto tmp81 = static_cast<long>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<long>(28);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<long>((-1L) + (2L*x3));
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp91 = [&]
                                {
                                    auto tmp92 = static_cast<float>(1.0);
                                    return tmp92;
                                }
                                ;
                                auto tmp93 = tmp90 ? tmp91() : static_cast<decltype(tmp91())>(1.0);
                                return tmp93;
                            }
                            ;
                            auto tmp94 = tmp78 ? tmp79() : static_cast<decltype(tmp79())>(0.0);
                            auto tmp95 = tmp14 >= tmp70;
                            auto tmp96 = tmp14 < tmp72;
                            auto tmp97 = tmp95 & tmp96;
                            auto tmp98 = tmp74 & tmp97;
                            auto tmp99 = [&]
                            {
                                auto tmp100 = c10::convert<long>((-1L) + (2L*x2));
                                auto tmp101 = static_cast<long>(0);
                                auto tmp102 = tmp100 >= tmp101;
                                auto tmp103 = static_cast<long>(28);
                                auto tmp104 = tmp100 < tmp103;
                                auto tmp105 = tmp102 & tmp104;
                                auto tmp106 = c10::convert<long>(2L*x3);
                                auto tmp107 = tmp106 >= tmp101;
                                auto tmp108 = tmp106 < tmp103;
                                auto tmp109 = tmp107 & tmp108;
                                auto tmp110 = tmp105 & tmp109;
                                auto tmp111 = [&]
                                {
                                    auto tmp112 = static_cast<float>(1.0);
                                    return tmp112;
                                }
                                ;
                                auto tmp113 = tmp110 ? tmp111() : static_cast<decltype(tmp111())>(1.0);
                                return tmp113;
                            }
                            ;
                            auto tmp114 = tmp98 ? tmp99() : static_cast<decltype(tmp99())>(0.0);
                            auto tmp115 = decltype(tmp114)(tmp114 + tmp94);
                            auto tmp116 = tmp23 >= tmp70;
                            auto tmp117 = tmp23 < tmp72;
                            auto tmp118 = tmp116 & tmp117;
                            auto tmp119 = tmp74 & tmp118;
                            auto tmp120 = [&]
                            {
                                auto tmp121 = c10::convert<long>((-1L) + (2L*x2));
                                auto tmp122 = static_cast<long>(0);
                                auto tmp123 = tmp121 >= tmp122;
                                auto tmp124 = static_cast<long>(28);
                                auto tmp125 = tmp121 < tmp124;
                                auto tmp126 = tmp123 & tmp125;
                                auto tmp127 = c10::convert<long>(1L + (2L*x3));
                                auto tmp128 = tmp127 >= tmp122;
                                auto tmp129 = tmp127 < tmp124;
                                auto tmp130 = tmp128 & tmp129;
                                auto tmp131 = tmp126 & tmp130;
                                auto tmp132 = [&]
                                {
                                    auto tmp133 = static_cast<float>(1.0);
                                    return tmp133;
                                }
                                ;
                                auto tmp134 = tmp131 ? tmp132() : static_cast<decltype(tmp132())>(1.0);
                                return tmp134;
                            }
                            ;
                            auto tmp135 = tmp119 ? tmp120() : static_cast<decltype(tmp120())>(0.0);
                            auto tmp136 = decltype(tmp135)(tmp135 + tmp115);
                            auto tmp137 = tmp32 >= tmp70;
                            auto tmp138 = tmp32 < tmp72;
                            auto tmp139 = tmp137 & tmp138;
                            auto tmp140 = tmp139 & tmp77;
                            auto tmp141 = [&]
                            {
                                auto tmp142 = c10::convert<long>(2L*x2);
                                auto tmp143 = static_cast<long>(0);
                                auto tmp144 = tmp142 >= tmp143;
                                auto tmp145 = static_cast<long>(28);
                                auto tmp146 = tmp142 < tmp145;
                                auto tmp147 = tmp144 & tmp146;
                                auto tmp148 = c10::convert<long>((-1L) + (2L*x3));
                                auto tmp149 = tmp148 >= tmp143;
                                auto tmp150 = tmp148 < tmp145;
                                auto tmp151 = tmp149 & tmp150;
                                auto tmp152 = tmp147 & tmp151;
                                auto tmp153 = [&]
                                {
                                    auto tmp154 = static_cast<float>(1.0);
                                    return tmp154;
                                }
                                ;
                                auto tmp155 = tmp152 ? tmp153() : static_cast<decltype(tmp153())>(1.0);
                                return tmp155;
                            }
                            ;
                            auto tmp156 = tmp140 ? tmp141() : static_cast<decltype(tmp141())>(0.0);
                            auto tmp157 = decltype(tmp156)(tmp156 + tmp136);
                            auto tmp158 = tmp139 & tmp97;
                            auto tmp159 = [&]
                            {
                                auto tmp160 = c10::convert<long>(2L*x2);
                                auto tmp161 = static_cast<long>(0);
                                auto tmp162 = tmp160 >= tmp161;
                                auto tmp163 = static_cast<long>(28);
                                auto tmp164 = tmp160 < tmp163;
                                auto tmp165 = tmp162 & tmp164;
                                auto tmp166 = c10::convert<long>(2L*x3);
                                auto tmp167 = tmp166 >= tmp161;
                                auto tmp168 = tmp166 < tmp163;
                                auto tmp169 = tmp167 & tmp168;
                                auto tmp170 = tmp165 & tmp169;
                                auto tmp171 = [&]
                                {
                                    auto tmp172 = static_cast<float>(1.0);
                                    return tmp172;
                                }
                                ;
                                auto tmp173 = tmp170 ? tmp171() : static_cast<decltype(tmp171())>(1.0);
                                return tmp173;
                            }
                            ;
                            auto tmp174 = tmp158 ? tmp159() : static_cast<decltype(tmp159())>(0.0);
                            auto tmp175 = decltype(tmp174)(tmp174 + tmp157);
                            auto tmp176 = tmp139 & tmp118;
                            auto tmp177 = [&]
                            {
                                auto tmp178 = c10::convert<long>(2L*x2);
                                auto tmp179 = static_cast<long>(0);
                                auto tmp180 = tmp178 >= tmp179;
                                auto tmp181 = static_cast<long>(28);
                                auto tmp182 = tmp178 < tmp181;
                                auto tmp183 = tmp180 & tmp182;
                                auto tmp184 = c10::convert<long>(1L + (2L*x3));
                                auto tmp185 = tmp184 >= tmp179;
                                auto tmp186 = tmp184 < tmp181;
                                auto tmp187 = tmp185 & tmp186;
                                auto tmp188 = tmp183 & tmp187;
                                auto tmp189 = [&]
                                {
                                    auto tmp190 = static_cast<float>(1.0);
                                    return tmp190;
                                }
                                ;
                                auto tmp191 = tmp188 ? tmp189() : static_cast<decltype(tmp189())>(1.0);
                                return tmp191;
                            }
                            ;
                            auto tmp192 = tmp176 ? tmp177() : static_cast<decltype(tmp177())>(0.0);
                            auto tmp193 = decltype(tmp192)(tmp192 + tmp175);
                            auto tmp194 = tmp51 >= tmp70;
                            auto tmp195 = tmp51 < tmp72;
                            auto tmp196 = tmp194 & tmp195;
                            auto tmp197 = tmp196 & tmp77;
                            auto tmp198 = [&]
                            {
                                auto tmp199 = c10::convert<long>(1L + (2L*x2));
                                auto tmp200 = static_cast<long>(0);
                                auto tmp201 = tmp199 >= tmp200;
                                auto tmp202 = static_cast<long>(28);
                                auto tmp203 = tmp199 < tmp202;
                                auto tmp204 = tmp201 & tmp203;
                                auto tmp205 = c10::convert<long>((-1L) + (2L*x3));
                                auto tmp206 = tmp205 >= tmp200;
                                auto tmp207 = tmp205 < tmp202;
                                auto tmp208 = tmp206 & tmp207;
                                auto tmp209 = tmp204 & tmp208;
                                auto tmp210 = [&]
                                {
                                    auto tmp211 = static_cast<float>(1.0);
                                    return tmp211;
                                }
                                ;
                                auto tmp212 = tmp209 ? tmp210() : static_cast<decltype(tmp210())>(1.0);
                                return tmp212;
                            }
                            ;
                            auto tmp213 = tmp197 ? tmp198() : static_cast<decltype(tmp198())>(0.0);
                            auto tmp214 = decltype(tmp213)(tmp213 + tmp193);
                            auto tmp215 = tmp196 & tmp97;
                            auto tmp216 = [&]
                            {
                                auto tmp217 = c10::convert<long>(1L + (2L*x2));
                                auto tmp218 = static_cast<long>(0);
                                auto tmp219 = tmp217 >= tmp218;
                                auto tmp220 = static_cast<long>(28);
                                auto tmp221 = tmp217 < tmp220;
                                auto tmp222 = tmp219 & tmp221;
                                auto tmp223 = c10::convert<long>(2L*x3);
                                auto tmp224 = tmp223 >= tmp218;
                                auto tmp225 = tmp223 < tmp220;
                                auto tmp226 = tmp224 & tmp225;
                                auto tmp227 = tmp222 & tmp226;
                                auto tmp228 = [&]
                                {
                                    auto tmp229 = static_cast<float>(1.0);
                                    return tmp229;
                                }
                                ;
                                auto tmp230 = tmp227 ? tmp228() : static_cast<decltype(tmp228())>(1.0);
                                return tmp230;
                            }
                            ;
                            auto tmp231 = tmp215 ? tmp216() : static_cast<decltype(tmp216())>(0.0);
                            auto tmp232 = decltype(tmp231)(tmp231 + tmp214);
                            auto tmp233 = tmp196 & tmp118;
                            auto tmp234 = [&]
                            {
                                auto tmp235 = c10::convert<long>(1L + (2L*x2));
                                auto tmp236 = static_cast<long>(0);
                                auto tmp237 = tmp235 >= tmp236;
                                auto tmp238 = static_cast<long>(28);
                                auto tmp239 = tmp235 < tmp238;
                                auto tmp240 = tmp237 & tmp239;
                                auto tmp241 = c10::convert<long>(1L + (2L*x3));
                                auto tmp242 = tmp241 >= tmp236;
                                auto tmp243 = tmp241 < tmp238;
                                auto tmp244 = tmp242 & tmp243;
                                auto tmp245 = tmp240 & tmp244;
                                auto tmp246 = [&]
                                {
                                    auto tmp247 = static_cast<float>(1.0);
                                    return tmp247;
                                }
                                ;
                                auto tmp248 = tmp245 ? tmp246() : static_cast<decltype(tmp246())>(1.0);
                                return tmp248;
                            }
                            ;
                            auto tmp249 = tmp233 ? tmp234() : static_cast<decltype(tmp234())>(0.0);
                            auto tmp250 = decltype(tmp249)(tmp249 + tmp232);
                            auto tmp251 = tmp69 / tmp250;
                            out_ptr0[static_cast<long>(x3 + (14L*x2) + (196L*x1) + (100352L*x0))] = tmp251;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr10[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr11 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr12[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr13[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr14[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr15[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr16[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_41 = async_compile.cpp('''
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
                       const float* in_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = tmp16 + tmp29;
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_43 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(128L + x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (25088L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr5[static_cast<long>(128L + x2 + (512L*x1) + (100352L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        out_ptr1[static_cast<long>(x2 + (128L*x1) + (25088L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_44 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(256L + x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (25088L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr5[static_cast<long>(256L + x2 + (512L*x1) + (100352L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        out_ptr1[static_cast<long>(x2 + (128L*x1) + (25088L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_45 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(384L + x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(384L + x1 + (512L*x2) + (100352L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_48 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(128L + x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (25088L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr5[static_cast<long>(128L + x2 + (512L*x1) + (100352L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        out_ptr1[static_cast<long>(x2 + (128L*x1) + (25088L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_49 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(256L + x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (25088L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr5[static_cast<long>(256L + x2 + (512L*x1) + (100352L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        out_ptr1[static_cast<long>(x2 + (128L*x1) + (25088L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_50 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(384L + x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(384L + x1 + (512L*x2) + (100352L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_53 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(128L + x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (25088L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr5[static_cast<long>(128L + x2 + (512L*x1) + (100352L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        out_ptr1[static_cast<long>(x2 + (128L*x1) + (25088L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_54 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(256L + x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (25088L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr5[static_cast<long>(256L + x2 + (512L*x1) + (100352L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        out_ptr1[static_cast<long>(x2 + (128L*x1) + (25088L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_55 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(384L + x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(384L + x1 + (512L*x2) + (100352L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_58 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(128L + x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (25088L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr5[static_cast<long>(128L + x2 + (512L*x1) + (100352L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        out_ptr1[static_cast<long>(x2 + (128L*x1) + (25088L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_59 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(256L + x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (25088L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr5[static_cast<long>(256L + x2 + (512L*x1) + (100352L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        out_ptr1[static_cast<long>(x2 + (128L*x1) + (25088L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_60 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(384L + x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(384L + x1 + (512L*x2) + (100352L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_63 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(128L + x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (25088L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr5[static_cast<long>(128L + x2 + (512L*x1) + (100352L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        out_ptr1[static_cast<long>(x2 + (128L*x1) + (25088L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_64 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(256L + x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (25088L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr5[static_cast<long>(256L + x2 + (512L*x1) + (100352L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        out_ptr1[static_cast<long>(x2 + (128L*x1) + (25088L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_65 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(384L + x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(384L + x1 + (512L*x2) + (100352L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_70 = async_compile.cpp('''
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
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-1L) + (2L*x2));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(14);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = c10::convert<long>((-1L) + (2L*x3));
                        auto tmp7 = tmp6 >= tmp1;
                        auto tmp8 = tmp6 < tmp3;
                        auto tmp9 = tmp7 & tmp8;
                        auto tmp10 = tmp5 & tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr0[static_cast<long>((-14592L) + x1 + (2048L*x3) + (28672L*x2) + (200704L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = c10::convert<long>(2L*x3);
                        auto tmp15 = tmp14 >= tmp1;
                        auto tmp16 = tmp14 < tmp3;
                        auto tmp17 = tmp15 & tmp16;
                        auto tmp18 = tmp5 & tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr0[static_cast<long>((-13568L) + x1 + (2048L*x3) + (28672L*x2) + (200704L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = decltype(tmp21)(tmp21 + tmp13);
                        auto tmp23 = c10::convert<long>(1L + (2L*x3));
                        auto tmp24 = tmp23 >= tmp1;
                        auto tmp25 = tmp23 < tmp3;
                        auto tmp26 = tmp24 & tmp25;
                        auto tmp27 = tmp5 & tmp26;
                        auto tmp28 = [&]
                        {
                            auto tmp29 = in_ptr0[static_cast<long>((-12544L) + x1 + (2048L*x3) + (28672L*x2) + (200704L*x0))];
                            return tmp29;
                        }
                        ;
                        auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                        auto tmp31 = decltype(tmp30)(tmp30 + tmp22);
                        auto tmp32 = c10::convert<long>(2L*x2);
                        auto tmp33 = tmp32 >= tmp1;
                        auto tmp34 = tmp32 < tmp3;
                        auto tmp35 = tmp33 & tmp34;
                        auto tmp36 = tmp35 & tmp9;
                        auto tmp37 = [&]
                        {
                            auto tmp38 = in_ptr0[static_cast<long>((-256L) + x1 + (2048L*x3) + (28672L*x2) + (200704L*x0))];
                            return tmp38;
                        }
                        ;
                        auto tmp39 = tmp36 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                        auto tmp40 = decltype(tmp39)(tmp39 + tmp31);
                        auto tmp41 = tmp35 & tmp17;
                        auto tmp42 = [&]
                        {
                            auto tmp43 = in_ptr0[static_cast<long>(768L + x1 + (2048L*x3) + (28672L*x2) + (200704L*x0))];
                            return tmp43;
                        }
                        ;
                        auto tmp44 = tmp41 ? tmp42() : static_cast<decltype(tmp42())>(0.0);
                        auto tmp45 = decltype(tmp44)(tmp44 + tmp40);
                        auto tmp46 = tmp35 & tmp26;
                        auto tmp47 = [&]
                        {
                            auto tmp48 = in_ptr0[static_cast<long>(1792L + x1 + (2048L*x3) + (28672L*x2) + (200704L*x0))];
                            return tmp48;
                        }
                        ;
                        auto tmp49 = tmp46 ? tmp47() : static_cast<decltype(tmp47())>(0.0);
                        auto tmp50 = decltype(tmp49)(tmp49 + tmp45);
                        auto tmp51 = c10::convert<long>(1L + (2L*x2));
                        auto tmp52 = tmp51 >= tmp1;
                        auto tmp53 = tmp51 < tmp3;
                        auto tmp54 = tmp52 & tmp53;
                        auto tmp55 = tmp54 & tmp9;
                        auto tmp56 = [&]
                        {
                            auto tmp57 = in_ptr0[static_cast<long>(14080L + x1 + (2048L*x3) + (28672L*x2) + (200704L*x0))];
                            return tmp57;
                        }
                        ;
                        auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                        auto tmp59 = decltype(tmp58)(tmp58 + tmp50);
                        auto tmp60 = tmp54 & tmp17;
                        auto tmp61 = [&]
                        {
                            auto tmp62 = in_ptr0[static_cast<long>(15104L + x1 + (2048L*x3) + (28672L*x2) + (200704L*x0))];
                            return tmp62;
                        }
                        ;
                        auto tmp63 = tmp60 ? tmp61() : static_cast<decltype(tmp61())>(0.0);
                        auto tmp64 = decltype(tmp63)(tmp63 + tmp59);
                        auto tmp65 = tmp54 & tmp26;
                        auto tmp66 = [&]
                        {
                            auto tmp67 = in_ptr0[static_cast<long>(16128L + x1 + (2048L*x3) + (28672L*x2) + (200704L*x0))];
                            return tmp67;
                        }
                        ;
                        auto tmp68 = tmp65 ? tmp66() : static_cast<decltype(tmp66())>(0.0);
                        auto tmp69 = decltype(tmp68)(tmp68 + tmp64);
                        auto tmp70 = static_cast<long>(-1);
                        auto tmp71 = tmp0 >= tmp70;
                        auto tmp72 = static_cast<long>(15);
                        auto tmp73 = tmp0 < tmp72;
                        auto tmp74 = tmp71 & tmp73;
                        auto tmp75 = tmp6 >= tmp70;
                        auto tmp76 = tmp6 < tmp72;
                        auto tmp77 = tmp75 & tmp76;
                        auto tmp78 = tmp74 & tmp77;
                        auto tmp79 = [&]
                        {
                            auto tmp80 = c10::convert<long>((-1L) + (2L*x2));
                            auto tmp81 = static_cast<long>(0);
                            auto tmp82 = tmp80 >= tmp81;
                            auto tmp83 = static_cast<long>(14);
                            auto tmp84 = tmp80 < tmp83;
                            auto tmp85 = tmp82 & tmp84;
                            auto tmp86 = c10::convert<long>((-1L) + (2L*x3));
                            auto tmp87 = tmp86 >= tmp81;
                            auto tmp88 = tmp86 < tmp83;
                            auto tmp89 = tmp87 & tmp88;
                            auto tmp90 = tmp85 & tmp89;
                            auto tmp91 = [&]
                            {
                                auto tmp92 = static_cast<float>(1.0);
                                return tmp92;
                            }
                            ;
                            auto tmp93 = tmp90 ? tmp91() : static_cast<decltype(tmp91())>(1.0);
                            return tmp93;
                        }
                        ;
                        auto tmp94 = tmp78 ? tmp79() : static_cast<decltype(tmp79())>(0.0);
                        auto tmp95 = tmp14 >= tmp70;
                        auto tmp96 = tmp14 < tmp72;
                        auto tmp97 = tmp95 & tmp96;
                        auto tmp98 = tmp74 & tmp97;
                        auto tmp99 = [&]
                        {
                            auto tmp100 = c10::convert<long>((-1L) + (2L*x2));
                            auto tmp101 = static_cast<long>(0);
                            auto tmp102 = tmp100 >= tmp101;
                            auto tmp103 = static_cast<long>(14);
                            auto tmp104 = tmp100 < tmp103;
                            auto tmp105 = tmp102 & tmp104;
                            auto tmp106 = c10::convert<long>(2L*x3);
                            auto tmp107 = tmp106 >= tmp101;
                            auto tmp108 = tmp106 < tmp103;
                            auto tmp109 = tmp107 & tmp108;
                            auto tmp110 = tmp105 & tmp109;
                            auto tmp111 = [&]
                            {
                                auto tmp112 = static_cast<float>(1.0);
                                return tmp112;
                            }
                            ;
                            auto tmp113 = tmp110 ? tmp111() : static_cast<decltype(tmp111())>(1.0);
                            return tmp113;
                        }
                        ;
                        auto tmp114 = tmp98 ? tmp99() : static_cast<decltype(tmp99())>(0.0);
                        auto tmp115 = decltype(tmp114)(tmp114 + tmp94);
                        auto tmp116 = tmp23 >= tmp70;
                        auto tmp117 = tmp23 < tmp72;
                        auto tmp118 = tmp116 & tmp117;
                        auto tmp119 = tmp74 & tmp118;
                        auto tmp120 = [&]
                        {
                            auto tmp121 = c10::convert<long>((-1L) + (2L*x2));
                            auto tmp122 = static_cast<long>(0);
                            auto tmp123 = tmp121 >= tmp122;
                            auto tmp124 = static_cast<long>(14);
                            auto tmp125 = tmp121 < tmp124;
                            auto tmp126 = tmp123 & tmp125;
                            auto tmp127 = c10::convert<long>(1L + (2L*x3));
                            auto tmp128 = tmp127 >= tmp122;
                            auto tmp129 = tmp127 < tmp124;
                            auto tmp130 = tmp128 & tmp129;
                            auto tmp131 = tmp126 & tmp130;
                            auto tmp132 = [&]
                            {
                                auto tmp133 = static_cast<float>(1.0);
                                return tmp133;
                            }
                            ;
                            auto tmp134 = tmp131 ? tmp132() : static_cast<decltype(tmp132())>(1.0);
                            return tmp134;
                        }
                        ;
                        auto tmp135 = tmp119 ? tmp120() : static_cast<decltype(tmp120())>(0.0);
                        auto tmp136 = decltype(tmp135)(tmp135 + tmp115);
                        auto tmp137 = tmp32 >= tmp70;
                        auto tmp138 = tmp32 < tmp72;
                        auto tmp139 = tmp137 & tmp138;
                        auto tmp140 = tmp139 & tmp77;
                        auto tmp141 = [&]
                        {
                            auto tmp142 = c10::convert<long>(2L*x2);
                            auto tmp143 = static_cast<long>(0);
                            auto tmp144 = tmp142 >= tmp143;
                            auto tmp145 = static_cast<long>(14);
                            auto tmp146 = tmp142 < tmp145;
                            auto tmp147 = tmp144 & tmp146;
                            auto tmp148 = c10::convert<long>((-1L) + (2L*x3));
                            auto tmp149 = tmp148 >= tmp143;
                            auto tmp150 = tmp148 < tmp145;
                            auto tmp151 = tmp149 & tmp150;
                            auto tmp152 = tmp147 & tmp151;
                            auto tmp153 = [&]
                            {
                                auto tmp154 = static_cast<float>(1.0);
                                return tmp154;
                            }
                            ;
                            auto tmp155 = tmp152 ? tmp153() : static_cast<decltype(tmp153())>(1.0);
                            return tmp155;
                        }
                        ;
                        auto tmp156 = tmp140 ? tmp141() : static_cast<decltype(tmp141())>(0.0);
                        auto tmp157 = decltype(tmp156)(tmp156 + tmp136);
                        auto tmp158 = tmp139 & tmp97;
                        auto tmp159 = [&]
                        {
                            auto tmp160 = c10::convert<long>(2L*x2);
                            auto tmp161 = static_cast<long>(0);
                            auto tmp162 = tmp160 >= tmp161;
                            auto tmp163 = static_cast<long>(14);
                            auto tmp164 = tmp160 < tmp163;
                            auto tmp165 = tmp162 & tmp164;
                            auto tmp166 = c10::convert<long>(2L*x3);
                            auto tmp167 = tmp166 >= tmp161;
                            auto tmp168 = tmp166 < tmp163;
                            auto tmp169 = tmp167 & tmp168;
                            auto tmp170 = tmp165 & tmp169;
                            auto tmp171 = [&]
                            {
                                auto tmp172 = static_cast<float>(1.0);
                                return tmp172;
                            }
                            ;
                            auto tmp173 = tmp170 ? tmp171() : static_cast<decltype(tmp171())>(1.0);
                            return tmp173;
                        }
                        ;
                        auto tmp174 = tmp158 ? tmp159() : static_cast<decltype(tmp159())>(0.0);
                        auto tmp175 = decltype(tmp174)(tmp174 + tmp157);
                        auto tmp176 = tmp139 & tmp118;
                        auto tmp177 = [&]
                        {
                            auto tmp178 = c10::convert<long>(2L*x2);
                            auto tmp179 = static_cast<long>(0);
                            auto tmp180 = tmp178 >= tmp179;
                            auto tmp181 = static_cast<long>(14);
                            auto tmp182 = tmp178 < tmp181;
                            auto tmp183 = tmp180 & tmp182;
                            auto tmp184 = c10::convert<long>(1L + (2L*x3));
                            auto tmp185 = tmp184 >= tmp179;
                            auto tmp186 = tmp184 < tmp181;
                            auto tmp187 = tmp185 & tmp186;
                            auto tmp188 = tmp183 & tmp187;
                            auto tmp189 = [&]
                            {
                                auto tmp190 = static_cast<float>(1.0);
                                return tmp190;
                            }
                            ;
                            auto tmp191 = tmp188 ? tmp189() : static_cast<decltype(tmp189())>(1.0);
                            return tmp191;
                        }
                        ;
                        auto tmp192 = tmp176 ? tmp177() : static_cast<decltype(tmp177())>(0.0);
                        auto tmp193 = decltype(tmp192)(tmp192 + tmp175);
                        auto tmp194 = tmp51 >= tmp70;
                        auto tmp195 = tmp51 < tmp72;
                        auto tmp196 = tmp194 & tmp195;
                        auto tmp197 = tmp196 & tmp77;
                        auto tmp198 = [&]
                        {
                            auto tmp199 = c10::convert<long>(1L + (2L*x2));
                            auto tmp200 = static_cast<long>(0);
                            auto tmp201 = tmp199 >= tmp200;
                            auto tmp202 = static_cast<long>(14);
                            auto tmp203 = tmp199 < tmp202;
                            auto tmp204 = tmp201 & tmp203;
                            auto tmp205 = c10::convert<long>((-1L) + (2L*x3));
                            auto tmp206 = tmp205 >= tmp200;
                            auto tmp207 = tmp205 < tmp202;
                            auto tmp208 = tmp206 & tmp207;
                            auto tmp209 = tmp204 & tmp208;
                            auto tmp210 = [&]
                            {
                                auto tmp211 = static_cast<float>(1.0);
                                return tmp211;
                            }
                            ;
                            auto tmp212 = tmp209 ? tmp210() : static_cast<decltype(tmp210())>(1.0);
                            return tmp212;
                        }
                        ;
                        auto tmp213 = tmp197 ? tmp198() : static_cast<decltype(tmp198())>(0.0);
                        auto tmp214 = decltype(tmp213)(tmp213 + tmp193);
                        auto tmp215 = tmp196 & tmp97;
                        auto tmp216 = [&]
                        {
                            auto tmp217 = c10::convert<long>(1L + (2L*x2));
                            auto tmp218 = static_cast<long>(0);
                            auto tmp219 = tmp217 >= tmp218;
                            auto tmp220 = static_cast<long>(14);
                            auto tmp221 = tmp217 < tmp220;
                            auto tmp222 = tmp219 & tmp221;
                            auto tmp223 = c10::convert<long>(2L*x3);
                            auto tmp224 = tmp223 >= tmp218;
                            auto tmp225 = tmp223 < tmp220;
                            auto tmp226 = tmp224 & tmp225;
                            auto tmp227 = tmp222 & tmp226;
                            auto tmp228 = [&]
                            {
                                auto tmp229 = static_cast<float>(1.0);
                                return tmp229;
                            }
                            ;
                            auto tmp230 = tmp227 ? tmp228() : static_cast<decltype(tmp228())>(1.0);
                            return tmp230;
                        }
                        ;
                        auto tmp231 = tmp215 ? tmp216() : static_cast<decltype(tmp216())>(0.0);
                        auto tmp232 = decltype(tmp231)(tmp231 + tmp214);
                        auto tmp233 = tmp196 & tmp118;
                        auto tmp234 = [&]
                        {
                            auto tmp235 = c10::convert<long>(1L + (2L*x2));
                            auto tmp236 = static_cast<long>(0);
                            auto tmp237 = tmp235 >= tmp236;
                            auto tmp238 = static_cast<long>(14);
                            auto tmp239 = tmp235 < tmp238;
                            auto tmp240 = tmp237 & tmp239;
                            auto tmp241 = c10::convert<long>(1L + (2L*x3));
                            auto tmp242 = tmp241 >= tmp236;
                            auto tmp243 = tmp241 < tmp238;
                            auto tmp244 = tmp242 & tmp243;
                            auto tmp245 = tmp240 & tmp244;
                            auto tmp246 = [&]
                            {
                                auto tmp247 = static_cast<float>(1.0);
                                return tmp247;
                            }
                            ;
                            auto tmp248 = tmp245 ? tmp246() : static_cast<decltype(tmp246())>(1.0);
                            return tmp248;
                        }
                        ;
                        auto tmp249 = tmp233 ? tmp234() : static_cast<decltype(tmp234())>(0.0);
                        auto tmp250 = decltype(tmp249)(tmp249 + tmp232);
                        auto tmp251 = tmp69 / tmp250;
                        out_ptr0[static_cast<long>(x3 + (7L*x2) + (49L*x1) + (50176L*x0))] = tmp251;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)), static_cast<long>(256L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp14 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp17 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = std::sqrt(tmp7);
                        auto tmp9 = 1 / tmp8;
                        auto tmp10 = static_cast<float>(1.0);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp4 * tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 + tmp18;
                        auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                        tmp20.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)), static_cast<long>(256L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                        auto tmp5 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                        auto tmp14 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                        auto tmp17 = in_ptr10[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = std::sqrt(tmp7);
                        auto tmp9 = 1 / tmp8;
                        auto tmp10 = static_cast<float>(1.0);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp4 * tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 + tmp18;
                        auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                        tmp20.store(out_ptr2 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr11 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)), static_cast<long>(256L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = in_ptr12[static_cast<long>(x1 + x1_inner)];
                        auto tmp5 = in_ptr13[static_cast<long>(x1 + x1_inner)];
                        auto tmp14 = in_ptr14[static_cast<long>(x1 + x1_inner)];
                        auto tmp17 = in_ptr15[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = std::sqrt(tmp7);
                        auto tmp9 = 1 / tmp8;
                        auto tmp10 = static_cast<float>(1.0);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp4 * tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 + tmp18;
                        auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                        tmp20.store(out_ptr3 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)), static_cast<long>(1024L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr16[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_71 = async_compile.cpp('''
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
                       const float* in_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = tmp16 + tmp29;
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_73 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)), static_cast<long>(256L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                        auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = std::sqrt(tmp7);
                        auto tmp9 = 1 / tmp8;
                        auto tmp10 = static_cast<float>(1.0);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp4 * tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 + tmp18;
                        auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                        tmp20.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (49L*x2) + (50176L*x0)), static_cast<long>(49L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(256L + x2 + (1024L*x1) + (1024L*x1_inner) + (50176L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (12544L*x0)));
                    }
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (49L*x2) + (50176L*x0))];
                    auto tmp1 = in_ptr5[static_cast<long>(256L + x2 + (1024L*x1) + (50176L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    out_ptr1[static_cast<long>(x2 + (256L*x1) + (12544L*x0))] = tmp2;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_74 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)), static_cast<long>(256L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                        auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = std::sqrt(tmp7);
                        auto tmp9 = 1 / tmp8;
                        auto tmp10 = static_cast<float>(1.0);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp4 * tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 + tmp18;
                        auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                        tmp20.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (49L*x2) + (50176L*x0)), static_cast<long>(49L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(512L + x2 + (1024L*x1) + (1024L*x1_inner) + (50176L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (12544L*x0)));
                    }
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (49L*x2) + (50176L*x0))];
                    auto tmp1 = in_ptr5[static_cast<long>(512L + x2 + (1024L*x1) + (50176L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    out_ptr1[static_cast<long>(x2 + (256L*x1) + (12544L*x0))] = tmp2;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_75 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)), static_cast<long>(256L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                        auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = std::sqrt(tmp7);
                        auto tmp9 = 1 / tmp8;
                        auto tmp10 = static_cast<float>(1.0);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp4 * tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 + tmp18;
                        auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                        tmp20.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(768L + x1 + (1024L*x2) + (50176L*x0)), static_cast<long>(1024L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(768L + x1 + (1024L*x2) + (50176L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)), static_cast<long>(1024L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_78 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)), static_cast<long>(256L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                        auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = std::sqrt(tmp7);
                        auto tmp9 = 1 / tmp8;
                        auto tmp10 = static_cast<float>(1.0);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp4 * tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 + tmp18;
                        auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                        tmp20.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (49L*x2) + (50176L*x0)), static_cast<long>(49L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(256L + x2 + (1024L*x1) + (1024L*x1_inner) + (50176L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (12544L*x0)));
                    }
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (49L*x2) + (50176L*x0))];
                    auto tmp1 = in_ptr5[static_cast<long>(256L + x2 + (1024L*x1) + (50176L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    out_ptr1[static_cast<long>(x2 + (256L*x1) + (12544L*x0))] = tmp2;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_79 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)), static_cast<long>(256L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                        auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = std::sqrt(tmp7);
                        auto tmp9 = 1 / tmp8;
                        auto tmp10 = static_cast<float>(1.0);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp4 * tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 + tmp18;
                        auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                        tmp20.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (49L*x2) + (50176L*x0)), static_cast<long>(49L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(512L + x2 + (1024L*x1) + (1024L*x1_inner) + (50176L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (12544L*x0)));
                    }
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (49L*x2) + (50176L*x0))];
                    auto tmp1 = in_ptr5[static_cast<long>(512L + x2 + (1024L*x1) + (50176L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    out_ptr1[static_cast<long>(x2 + (256L*x1) + (12544L*x0))] = tmp2;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_80 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)), static_cast<long>(256L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                        auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = std::sqrt(tmp7);
                        auto tmp9 = 1 / tmp8;
                        auto tmp10 = static_cast<float>(1.0);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp4 * tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 + tmp18;
                        auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                        tmp20.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(768L + x1 + (1024L*x2) + (50176L*x0)), static_cast<long>(1024L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(768L + x1 + (1024L*x2) + (50176L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)), static_cast<long>(1024L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x2) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (2048L*x2) + (100352L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.sqrt();
                            auto tmp8 = tmp7.reciprocal();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 * tmp10;
                            auto tmp12 = tmp2 * tmp11;
                            auto tmp14 = tmp12 * tmp13;
                            auto tmp16 = tmp14 + tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (32, ), (1, ))
    assert_size_stride(arg9_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg10_1, (32, ), (1, ))
    assert_size_stride(arg11_1, (32, ), (1, ))
    assert_size_stride(arg12_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg13_1, (32, ), (1, ))
    assert_size_stride(arg14_1, (32, ), (1, ))
    assert_size_stride(arg15_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg25_1, (32, ), (1, ))
    assert_size_stride(arg26_1, (32, ), (1, ))
    assert_size_stride(arg27_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg28_1, (32, ), (1, ))
    assert_size_stride(arg29_1, (32, ), (1, ))
    assert_size_stride(arg30_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg31_1, (32, ), (1, ))
    assert_size_stride(arg32_1, (32, ), (1, ))
    assert_size_stride(arg33_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg40_1, (32, ), (1, ))
    assert_size_stride(arg41_1, (32, ), (1, ))
    assert_size_stride(arg42_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg43_1, (32, ), (1, ))
    assert_size_stride(arg44_1, (32, ), (1, ))
    assert_size_stride(arg45_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg46_1, (32, ), (1, ))
    assert_size_stride(arg47_1, (32, ), (1, ))
    assert_size_stride(arg48_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (64, ), (1, ))
    assert_size_stride(arg57_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg58_1, (64, ), (1, ))
    assert_size_stride(arg59_1, (64, ), (1, ))
    assert_size_stride(arg60_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg61_1, (64, ), (1, ))
    assert_size_stride(arg62_1, (64, ), (1, ))
    assert_size_stride(arg63_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg70_1, (256, ), (1, ))
    assert_size_stride(arg71_1, (256, ), (1, ))
    assert_size_stride(arg72_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg73_1, (64, ), (1, ))
    assert_size_stride(arg74_1, (64, ), (1, ))
    assert_size_stride(arg75_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg76_1, (64, ), (1, ))
    assert_size_stride(arg77_1, (64, ), (1, ))
    assert_size_stride(arg78_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg79_1, (64, ), (1, ))
    assert_size_stride(arg80_1, (64, ), (1, ))
    assert_size_stride(arg81_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg88_1, (64, ), (1, ))
    assert_size_stride(arg89_1, (64, ), (1, ))
    assert_size_stride(arg90_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg91_1, (64, ), (1, ))
    assert_size_stride(arg92_1, (64, ), (1, ))
    assert_size_stride(arg93_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg94_1, (64, ), (1, ))
    assert_size_stride(arg95_1, (64, ), (1, ))
    assert_size_stride(arg96_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg103_1, (64, ), (1, ))
    assert_size_stride(arg104_1, (64, ), (1, ))
    assert_size_stride(arg105_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg106_1, (64, ), (1, ))
    assert_size_stride(arg107_1, (64, ), (1, ))
    assert_size_stride(arg108_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg109_1, (64, ), (1, ))
    assert_size_stride(arg110_1, (64, ), (1, ))
    assert_size_stride(arg111_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (512, ), (1, ))
    assert_size_stride(arg117_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg124_1, (128, ), (1, ))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (512, ), (1, ))
    assert_size_stride(arg135_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg136_1, (128, ), (1, ))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg142_1, (128, ), (1, ))
    assert_size_stride(arg143_1, (128, ), (1, ))
    assert_size_stride(arg144_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg148_1, (512, ), (1, ))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg151_1, (128, ), (1, ))
    assert_size_stride(arg152_1, (128, ), (1, ))
    assert_size_stride(arg153_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg154_1, (128, ), (1, ))
    assert_size_stride(arg155_1, (128, ), (1, ))
    assert_size_stride(arg156_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg157_1, (128, ), (1, ))
    assert_size_stride(arg158_1, (128, ), (1, ))
    assert_size_stride(arg159_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg160_1, (1024, ), (1, ))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg163_1, (512, ), (1, ))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg166_1, (128, ), (1, ))
    assert_size_stride(arg167_1, (128, ), (1, ))
    assert_size_stride(arg168_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg169_1, (128, ), (1, ))
    assert_size_stride(arg170_1, (128, ), (1, ))
    assert_size_stride(arg171_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg172_1, (128, ), (1, ))
    assert_size_stride(arg173_1, (128, ), (1, ))
    assert_size_stride(arg174_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (512, ), (1, ))
    assert_size_stride(arg180_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg181_1, (128, ), (1, ))
    assert_size_stride(arg182_1, (128, ), (1, ))
    assert_size_stride(arg183_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg184_1, (128, ), (1, ))
    assert_size_stride(arg185_1, (128, ), (1, ))
    assert_size_stride(arg186_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg187_1, (128, ), (1, ))
    assert_size_stride(arg188_1, (128, ), (1, ))
    assert_size_stride(arg189_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg193_1, (512, ), (1, ))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg196_1, (128, ), (1, ))
    assert_size_stride(arg197_1, (128, ), (1, ))
    assert_size_stride(arg198_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg199_1, (128, ), (1, ))
    assert_size_stride(arg200_1, (128, ), (1, ))
    assert_size_stride(arg201_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg202_1, (128, ), (1, ))
    assert_size_stride(arg203_1, (128, ), (1, ))
    assert_size_stride(arg204_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg205_1, (1024, ), (1, ))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg208_1, (1024, ), (1, ))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg211_1, (256, ), (1, ))
    assert_size_stride(arg212_1, (256, ), (1, ))
    assert_size_stride(arg213_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg214_1, (256, ), (1, ))
    assert_size_stride(arg215_1, (256, ), (1, ))
    assert_size_stride(arg216_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg217_1, (256, ), (1, ))
    assert_size_stride(arg218_1, (256, ), (1, ))
    assert_size_stride(arg219_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg220_1, (2048, ), (1, ))
    assert_size_stride(arg221_1, (2048, ), (1, ))
    assert_size_stride(arg222_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg223_1, (2048, ), (1, ))
    assert_size_stride(arg224_1, (2048, ), (1, ))
    assert_size_stride(arg225_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg229_1, (256, ), (1, ))
    assert_size_stride(arg230_1, (256, ), (1, ))
    assert_size_stride(arg231_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg232_1, (256, ), (1, ))
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg235_1, (256, ), (1, ))
    assert_size_stride(arg236_1, (256, ), (1, ))
    assert_size_stride(arg237_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg238_1, (2048, ), (1, ))
    assert_size_stride(arg239_1, (2048, ), (1, ))
    assert_size_stride(arg240_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg244_1, (256, ), (1, ))
    assert_size_stride(arg245_1, (256, ), (1, ))
    assert_size_stride(arg246_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg247_1, (256, ), (1, ))
    assert_size_stride(arg248_1, (256, ), (1, ))
    assert_size_stride(arg249_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg250_1, (256, ), (1, ))
    assert_size_stride(arg251_1, (256, ), (1, ))
    assert_size_stride(arg252_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg253_1, (2048, ), (1, ))
    assert_size_stride(arg254_1, (2048, ), (1, ))
    assert_size_stride(arg255_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg256_1, (1000, ), (1, ))
    assert_size_stride(arg257_1, (64, ), (1, ))
    assert_size_stride(arg258_1, (64, ), (1, ))
    assert_size_stride(arg259_1, (), ())
    assert_size_stride(arg260_1, (128, ), (1, ))
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (), ())
    assert_size_stride(arg263_1, (32, ), (1, ))
    assert_size_stride(arg264_1, (32, ), (1, ))
    assert_size_stride(arg265_1, (), ())
    assert_size_stride(arg266_1, (32, ), (1, ))
    assert_size_stride(arg267_1, (32, ), (1, ))
    assert_size_stride(arg268_1, (), ())
    assert_size_stride(arg269_1, (32, ), (1, ))
    assert_size_stride(arg270_1, (32, ), (1, ))
    assert_size_stride(arg271_1, (), ())
    assert_size_stride(arg272_1, (256, ), (1, ))
    assert_size_stride(arg273_1, (256, ), (1, ))
    assert_size_stride(arg274_1, (), ())
    assert_size_stride(arg275_1, (256, ), (1, ))
    assert_size_stride(arg276_1, (256, ), (1, ))
    assert_size_stride(arg277_1, (), ())
    assert_size_stride(arg278_1, (128, ), (1, ))
    assert_size_stride(arg279_1, (128, ), (1, ))
    assert_size_stride(arg280_1, (), ())
    assert_size_stride(arg281_1, (32, ), (1, ))
    assert_size_stride(arg282_1, (32, ), (1, ))
    assert_size_stride(arg283_1, (), ())
    assert_size_stride(arg284_1, (32, ), (1, ))
    assert_size_stride(arg285_1, (32, ), (1, ))
    assert_size_stride(arg286_1, (), ())
    assert_size_stride(arg287_1, (32, ), (1, ))
    assert_size_stride(arg288_1, (32, ), (1, ))
    assert_size_stride(arg289_1, (), ())
    assert_size_stride(arg290_1, (256, ), (1, ))
    assert_size_stride(arg291_1, (256, ), (1, ))
    assert_size_stride(arg292_1, (), ())
    assert_size_stride(arg293_1, (128, ), (1, ))
    assert_size_stride(arg294_1, (128, ), (1, ))
    assert_size_stride(arg295_1, (), ())
    assert_size_stride(arg296_1, (32, ), (1, ))
    assert_size_stride(arg297_1, (32, ), (1, ))
    assert_size_stride(arg298_1, (), ())
    assert_size_stride(arg299_1, (32, ), (1, ))
    assert_size_stride(arg300_1, (32, ), (1, ))
    assert_size_stride(arg301_1, (), ())
    assert_size_stride(arg302_1, (32, ), (1, ))
    assert_size_stride(arg303_1, (32, ), (1, ))
    assert_size_stride(arg304_1, (), ())
    assert_size_stride(arg305_1, (256, ), (1, ))
    assert_size_stride(arg306_1, (256, ), (1, ))
    assert_size_stride(arg307_1, (), ())
    assert_size_stride(arg308_1, (256, ), (1, ))
    assert_size_stride(arg309_1, (256, ), (1, ))
    assert_size_stride(arg310_1, (), ())
    assert_size_stride(arg311_1, (64, ), (1, ))
    assert_size_stride(arg312_1, (64, ), (1, ))
    assert_size_stride(arg313_1, (), ())
    assert_size_stride(arg314_1, (64, ), (1, ))
    assert_size_stride(arg315_1, (64, ), (1, ))
    assert_size_stride(arg316_1, (), ())
    assert_size_stride(arg317_1, (64, ), (1, ))
    assert_size_stride(arg318_1, (64, ), (1, ))
    assert_size_stride(arg319_1, (), ())
    assert_size_stride(arg320_1, (512, ), (1, ))
    assert_size_stride(arg321_1, (512, ), (1, ))
    assert_size_stride(arg322_1, (), ())
    assert_size_stride(arg323_1, (512, ), (1, ))
    assert_size_stride(arg324_1, (512, ), (1, ))
    assert_size_stride(arg325_1, (), ())
    assert_size_stride(arg326_1, (256, ), (1, ))
    assert_size_stride(arg327_1, (256, ), (1, ))
    assert_size_stride(arg328_1, (), ())
    assert_size_stride(arg329_1, (64, ), (1, ))
    assert_size_stride(arg330_1, (64, ), (1, ))
    assert_size_stride(arg331_1, (), ())
    assert_size_stride(arg332_1, (64, ), (1, ))
    assert_size_stride(arg333_1, (64, ), (1, ))
    assert_size_stride(arg334_1, (), ())
    assert_size_stride(arg335_1, (64, ), (1, ))
    assert_size_stride(arg336_1, (64, ), (1, ))
    assert_size_stride(arg337_1, (), ())
    assert_size_stride(arg338_1, (512, ), (1, ))
    assert_size_stride(arg339_1, (512, ), (1, ))
    assert_size_stride(arg340_1, (), ())
    assert_size_stride(arg341_1, (256, ), (1, ))
    assert_size_stride(arg342_1, (256, ), (1, ))
    assert_size_stride(arg343_1, (), ())
    assert_size_stride(arg344_1, (64, ), (1, ))
    assert_size_stride(arg345_1, (64, ), (1, ))
    assert_size_stride(arg346_1, (), ())
    assert_size_stride(arg347_1, (64, ), (1, ))
    assert_size_stride(arg348_1, (64, ), (1, ))
    assert_size_stride(arg349_1, (), ())
    assert_size_stride(arg350_1, (64, ), (1, ))
    assert_size_stride(arg351_1, (64, ), (1, ))
    assert_size_stride(arg352_1, (), ())
    assert_size_stride(arg353_1, (512, ), (1, ))
    assert_size_stride(arg354_1, (512, ), (1, ))
    assert_size_stride(arg355_1, (), ())
    assert_size_stride(arg356_1, (256, ), (1, ))
    assert_size_stride(arg357_1, (256, ), (1, ))
    assert_size_stride(arg358_1, (), ())
    assert_size_stride(arg359_1, (64, ), (1, ))
    assert_size_stride(arg360_1, (64, ), (1, ))
    assert_size_stride(arg361_1, (), ())
    assert_size_stride(arg362_1, (64, ), (1, ))
    assert_size_stride(arg363_1, (64, ), (1, ))
    assert_size_stride(arg364_1, (), ())
    assert_size_stride(arg365_1, (64, ), (1, ))
    assert_size_stride(arg366_1, (64, ), (1, ))
    assert_size_stride(arg367_1, (), ())
    assert_size_stride(arg368_1, (512, ), (1, ))
    assert_size_stride(arg369_1, (512, ), (1, ))
    assert_size_stride(arg370_1, (), ())
    assert_size_stride(arg371_1, (512, ), (1, ))
    assert_size_stride(arg372_1, (512, ), (1, ))
    assert_size_stride(arg373_1, (), ())
    assert_size_stride(arg374_1, (128, ), (1, ))
    assert_size_stride(arg375_1, (128, ), (1, ))
    assert_size_stride(arg376_1, (), ())
    assert_size_stride(arg377_1, (128, ), (1, ))
    assert_size_stride(arg378_1, (128, ), (1, ))
    assert_size_stride(arg379_1, (), ())
    assert_size_stride(arg380_1, (128, ), (1, ))
    assert_size_stride(arg381_1, (128, ), (1, ))
    assert_size_stride(arg382_1, (), ())
    assert_size_stride(arg383_1, (1024, ), (1, ))
    assert_size_stride(arg384_1, (1024, ), (1, ))
    assert_size_stride(arg385_1, (), ())
    assert_size_stride(arg386_1, (1024, ), (1, ))
    assert_size_stride(arg387_1, (1024, ), (1, ))
    assert_size_stride(arg388_1, (), ())
    assert_size_stride(arg389_1, (512, ), (1, ))
    assert_size_stride(arg390_1, (512, ), (1, ))
    assert_size_stride(arg391_1, (), ())
    assert_size_stride(arg392_1, (128, ), (1, ))
    assert_size_stride(arg393_1, (128, ), (1, ))
    assert_size_stride(arg394_1, (), ())
    assert_size_stride(arg395_1, (128, ), (1, ))
    assert_size_stride(arg396_1, (128, ), (1, ))
    assert_size_stride(arg397_1, (), ())
    assert_size_stride(arg398_1, (128, ), (1, ))
    assert_size_stride(arg399_1, (128, ), (1, ))
    assert_size_stride(arg400_1, (), ())
    assert_size_stride(arg401_1, (1024, ), (1, ))
    assert_size_stride(arg402_1, (1024, ), (1, ))
    assert_size_stride(arg403_1, (), ())
    assert_size_stride(arg404_1, (512, ), (1, ))
    assert_size_stride(arg405_1, (512, ), (1, ))
    assert_size_stride(arg406_1, (), ())
    assert_size_stride(arg407_1, (128, ), (1, ))
    assert_size_stride(arg408_1, (128, ), (1, ))
    assert_size_stride(arg409_1, (), ())
    assert_size_stride(arg410_1, (128, ), (1, ))
    assert_size_stride(arg411_1, (128, ), (1, ))
    assert_size_stride(arg412_1, (), ())
    assert_size_stride(arg413_1, (128, ), (1, ))
    assert_size_stride(arg414_1, (128, ), (1, ))
    assert_size_stride(arg415_1, (), ())
    assert_size_stride(arg416_1, (1024, ), (1, ))
    assert_size_stride(arg417_1, (1024, ), (1, ))
    assert_size_stride(arg418_1, (), ())
    assert_size_stride(arg419_1, (512, ), (1, ))
    assert_size_stride(arg420_1, (512, ), (1, ))
    assert_size_stride(arg421_1, (), ())
    assert_size_stride(arg422_1, (128, ), (1, ))
    assert_size_stride(arg423_1, (128, ), (1, ))
    assert_size_stride(arg424_1, (), ())
    assert_size_stride(arg425_1, (128, ), (1, ))
    assert_size_stride(arg426_1, (128, ), (1, ))
    assert_size_stride(arg427_1, (), ())
    assert_size_stride(arg428_1, (128, ), (1, ))
    assert_size_stride(arg429_1, (128, ), (1, ))
    assert_size_stride(arg430_1, (), ())
    assert_size_stride(arg431_1, (1024, ), (1, ))
    assert_size_stride(arg432_1, (1024, ), (1, ))
    assert_size_stride(arg433_1, (), ())
    assert_size_stride(arg434_1, (512, ), (1, ))
    assert_size_stride(arg435_1, (512, ), (1, ))
    assert_size_stride(arg436_1, (), ())
    assert_size_stride(arg437_1, (128, ), (1, ))
    assert_size_stride(arg438_1, (128, ), (1, ))
    assert_size_stride(arg439_1, (), ())
    assert_size_stride(arg440_1, (128, ), (1, ))
    assert_size_stride(arg441_1, (128, ), (1, ))
    assert_size_stride(arg442_1, (), ())
    assert_size_stride(arg443_1, (128, ), (1, ))
    assert_size_stride(arg444_1, (128, ), (1, ))
    assert_size_stride(arg445_1, (), ())
    assert_size_stride(arg446_1, (1024, ), (1, ))
    assert_size_stride(arg447_1, (1024, ), (1, ))
    assert_size_stride(arg448_1, (), ())
    assert_size_stride(arg449_1, (512, ), (1, ))
    assert_size_stride(arg450_1, (512, ), (1, ))
    assert_size_stride(arg451_1, (), ())
    assert_size_stride(arg452_1, (128, ), (1, ))
    assert_size_stride(arg453_1, (128, ), (1, ))
    assert_size_stride(arg454_1, (), ())
    assert_size_stride(arg455_1, (128, ), (1, ))
    assert_size_stride(arg456_1, (128, ), (1, ))
    assert_size_stride(arg457_1, (), ())
    assert_size_stride(arg458_1, (128, ), (1, ))
    assert_size_stride(arg459_1, (128, ), (1, ))
    assert_size_stride(arg460_1, (), ())
    assert_size_stride(arg461_1, (1024, ), (1, ))
    assert_size_stride(arg462_1, (1024, ), (1, ))
    assert_size_stride(arg463_1, (), ())
    assert_size_stride(arg464_1, (1024, ), (1, ))
    assert_size_stride(arg465_1, (1024, ), (1, ))
    assert_size_stride(arg466_1, (), ())
    assert_size_stride(arg467_1, (256, ), (1, ))
    assert_size_stride(arg468_1, (256, ), (1, ))
    assert_size_stride(arg469_1, (), ())
    assert_size_stride(arg470_1, (256, ), (1, ))
    assert_size_stride(arg471_1, (256, ), (1, ))
    assert_size_stride(arg472_1, (), ())
    assert_size_stride(arg473_1, (256, ), (1, ))
    assert_size_stride(arg474_1, (256, ), (1, ))
    assert_size_stride(arg475_1, (), ())
    assert_size_stride(arg476_1, (2048, ), (1, ))
    assert_size_stride(arg477_1, (2048, ), (1, ))
    assert_size_stride(arg478_1, (), ())
    assert_size_stride(arg479_1, (2048, ), (1, ))
    assert_size_stride(arg480_1, (2048, ), (1, ))
    assert_size_stride(arg481_1, (), ())
    assert_size_stride(arg482_1, (1024, ), (1, ))
    assert_size_stride(arg483_1, (1024, ), (1, ))
    assert_size_stride(arg484_1, (), ())
    assert_size_stride(arg485_1, (256, ), (1, ))
    assert_size_stride(arg486_1, (256, ), (1, ))
    assert_size_stride(arg487_1, (), ())
    assert_size_stride(arg488_1, (256, ), (1, ))
    assert_size_stride(arg489_1, (256, ), (1, ))
    assert_size_stride(arg490_1, (), ())
    assert_size_stride(arg491_1, (256, ), (1, ))
    assert_size_stride(arg492_1, (256, ), (1, ))
    assert_size_stride(arg493_1, (), ())
    assert_size_stride(arg494_1, (2048, ), (1, ))
    assert_size_stride(arg495_1, (2048, ), (1, ))
    assert_size_stride(arg496_1, (), ())
    assert_size_stride(arg497_1, (1024, ), (1, ))
    assert_size_stride(arg498_1, (1024, ), (1, ))
    assert_size_stride(arg499_1, (), ())
    assert_size_stride(arg500_1, (256, ), (1, ))
    assert_size_stride(arg501_1, (256, ), (1, ))
    assert_size_stride(arg502_1, (), ())
    assert_size_stride(arg503_1, (256, ), (1, ))
    assert_size_stride(arg504_1, (256, ), (1, ))
    assert_size_stride(arg505_1, (), ())
    assert_size_stride(arg506_1, (256, ), (1, ))
    assert_size_stride(arg507_1, (256, ), (1, ))
    assert_size_stride(arg508_1, (), ())
    assert_size_stride(arg509_1, (2048, ), (1, ))
    assert_size_stride(arg510_1, (2048, ), (1, ))
    assert_size_stride(arg511_1, (), ())
    assert_size_stride(arg512_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg512_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg512_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 64, 112, 112), (802816, 1, 7168, 64))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg1_1
    del arg257_1
    del arg258_1
    del arg2_1
    del buf3
    # Source Nodes: [out], Original ATen: [aten.convolution]
    buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf5, (8, 128, 56, 56), (401408, 1, 7168, 128))
    del arg3_1
    buf6 = buf5; del buf5  # reuse
    buf7 = empty_strided((32, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2(c_void_p(buf6.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg260_1
    del arg261_1
    del arg4_1
    del arg5_1
    del arg6_1
    # Source Nodes: [sp_1], Original ATen: [aten.convolution]
    buf8 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 32, 56, 56), (401408, 1, 7168, 128), 0), buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf8, (8, 32, 56, 56), (100352, 1, 1792, 32))
    buf9 = buf7; del buf7  # reuse
    cpp_fused_convolution_3(c_void_p(arg9_1.data_ptr()), c_void_p(buf9.data_ptr()))
    del arg9_1
    # Source Nodes: [sp_5], Original ATen: [aten.convolution]
    buf10 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 32, 56, 56), (401408, 1, 7168, 128), 32), buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf10, (8, 32, 56, 56), (100352, 1, 1792, 32))
    buf11 = buf9; del buf9  # reuse
    cpp_fused_convolution_4(c_void_p(arg12_1.data_ptr()), c_void_p(buf11.data_ptr()))
    del arg12_1
    # Source Nodes: [sp_9], Original ATen: [aten.convolution]
    buf12 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 32, 56, 56), (401408, 1, 7168, 128), 64), buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf12, (8, 32, 56, 56), (100352, 1, 1792, 32))
    buf17 = empty((8, 128, 56, 56), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf17, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
    buf14 = reinterpret_tensor(buf17, (8, 32, 56, 56), (401408, 3136, 56, 1), 0)  # alias
    buf15 = reinterpret_tensor(buf17, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
    buf16 = reinterpret_tensor(buf17, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
    buf18 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_5(c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()))
    del arg10_1
    del arg11_1
    del arg13_1
    del arg14_1
    del arg263_1
    del arg264_1
    del arg266_1
    del arg267_1
    del arg269_1
    del arg270_1
    del arg7_1
    del arg8_1
    del buf10
    del buf12
    del buf13
    del buf14
    del buf15
    del buf16
    del buf17
    # Source Nodes: [out_4], Original ATen: [aten.convolution]
    buf19 = extern_kernels.convolution(buf18, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf19, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del arg15_1
    # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
    buf20 = extern_kernels.convolution(buf4, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf20, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del arg18_1
    buf21 = buf19; del buf19  # reuse
    buf22 = buf21; del buf21  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_6(c_void_p(buf22.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()))
    del arg16_1
    del arg17_1
    del arg19_1
    del arg20_1
    del arg272_1
    del arg273_1
    del arg275_1
    del arg276_1
    del buf20
    # Source Nodes: [out_8, shortcut_2], Original ATen: [aten.convolution, aten.relu]
    buf23 = extern_kernels.convolution(buf22, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf23, (8, 128, 56, 56), (401408, 1, 7168, 128))
    del arg21_1
    buf24 = buf23; del buf23  # reuse
    buf25 = buf11; del buf11  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_7(c_void_p(buf24.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf25.data_ptr()))
    del arg22_1
    del arg23_1
    del arg24_1
    del arg278_1
    del arg279_1
    # Source Nodes: [sp_14], Original ATen: [aten.convolution]
    buf26 = extern_kernels.convolution(reinterpret_tensor(buf24, (8, 32, 56, 56), (401408, 1, 7168, 128), 0), buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf26, (8, 32, 56, 56), (100352, 1, 1792, 32))
    buf37 = reinterpret_tensor(buf18, (8, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf18  # reuse
    buf27 = reinterpret_tensor(buf37, (8, 32, 56, 56), (401408, 3136, 56, 1), 0)  # alias
    buf28 = buf8; del buf8  # reuse
    buf29 = buf25; del buf25  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_8(c_void_p(buf26.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    del arg25_1
    del arg26_1
    del arg27_1
    del arg281_1
    del arg282_1
    del buf26
    # Source Nodes: [sp_17, sp_18], Original ATen: [aten.add, aten.convolution]
    buf30 = extern_kernels.convolution(buf28, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf30, (8, 32, 56, 56), (100352, 1, 1792, 32))
    buf31 = reinterpret_tensor(buf37, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
    buf32 = buf28; del buf28  # reuse
    buf33 = buf29; del buf29  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_9(c_void_p(buf30.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()))
    del arg284_1
    del arg285_1
    del arg28_1
    del arg29_1
    del arg30_1
    del buf30
    # Source Nodes: [sp_21, sp_22], Original ATen: [aten.add, aten.convolution]
    buf34 = extern_kernels.convolution(buf32, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf34, (8, 32, 56, 56), (100352, 1, 1792, 32))
    del buf32
    buf35 = reinterpret_tensor(buf37, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
    buf36 = reinterpret_tensor(buf37, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
    buf38 = buf6; del buf6  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_10(c_void_p(buf34.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()))
    del arg287_1
    del arg288_1
    del arg31_1
    del arg32_1
    del buf24
    del buf27
    del buf31
    del buf35
    del buf36
    # Source Nodes: [out_12], Original ATen: [aten.convolution]
    buf39 = extern_kernels.convolution(buf38, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf39, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del arg33_1
    buf40 = buf22; del buf22  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_11(c_void_p(buf40.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg290_1
    del arg291_1
    del arg34_1
    del arg35_1
    del buf39
    # Source Nodes: [out_16], Original ATen: [aten.convolution]
    buf41 = extern_kernels.convolution(buf40, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf41, (8, 128, 56, 56), (401408, 1, 7168, 128))
    del arg36_1
    buf42 = buf41; del buf41  # reuse
    buf43 = buf33; del buf33  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_12(c_void_p(buf42.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf43.data_ptr()))
    del arg293_1
    del arg294_1
    del arg37_1
    del arg38_1
    del arg39_1
    # Source Nodes: [sp_27], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(reinterpret_tensor(buf42, (8, 32, 56, 56), (401408, 1, 7168, 128), 0), buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf44, (8, 32, 56, 56), (100352, 1, 1792, 32))
    buf55 = reinterpret_tensor(buf38, (8, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf38  # reuse
    buf45 = reinterpret_tensor(buf55, (8, 32, 56, 56), (401408, 3136, 56, 1), 0)  # alias
    buf46 = buf34; del buf34  # reuse
    buf47 = buf43; del buf43  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_13(c_void_p(buf44.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    del arg296_1
    del arg297_1
    del arg40_1
    del arg41_1
    del arg42_1
    del buf44
    # Source Nodes: [sp_30, sp_31], Original ATen: [aten.add, aten.convolution]
    buf48 = extern_kernels.convolution(buf46, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf48, (8, 32, 56, 56), (100352, 1, 1792, 32))
    buf49 = reinterpret_tensor(buf55, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
    buf50 = buf46; del buf46  # reuse
    buf51 = buf47; del buf47  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_14(c_void_p(buf48.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()))
    del arg299_1
    del arg300_1
    del arg43_1
    del arg44_1
    del arg45_1
    del buf48
    # Source Nodes: [sp_34, sp_35], Original ATen: [aten.add, aten.convolution]
    buf52 = extern_kernels.convolution(buf50, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf52, (8, 32, 56, 56), (100352, 1, 1792, 32))
    del buf51
    buf53 = reinterpret_tensor(buf55, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
    buf54 = reinterpret_tensor(buf55, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
    buf56 = reinterpret_tensor(buf37, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf37  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_15(c_void_p(buf52.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()))
    del arg302_1
    del arg303_1
    del arg46_1
    del arg47_1
    del buf42
    del buf45
    del buf49
    del buf53
    del buf54
    del buf55
    # Source Nodes: [out_20], Original ATen: [aten.convolution]
    buf57 = extern_kernels.convolution(buf56, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf57, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del arg48_1
    del buf56
    buf58 = buf40; del buf40  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_16(c_void_p(buf58.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg50_1.data_ptr()))
    del arg305_1
    del arg306_1
    del arg49_1
    del arg50_1
    del buf57
    # Source Nodes: [out_24], Original ATen: [aten.convolution]
    buf59 = extern_kernels.convolution(buf58, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf59, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del arg51_1
    buf60 = buf59; del buf59  # reuse
    buf61 = empty_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_17(c_void_p(buf60.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(buf61.data_ptr()))
    del arg308_1
    del arg309_1
    del arg52_1
    del arg53_1
    del arg54_1
    # Source Nodes: [sp_40], Original ATen: [aten.convolution]
    buf62 = extern_kernels.convolution(reinterpret_tensor(buf60, (8, 64, 56, 56), (802816, 1, 14336, 256), 0), buf61, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf62, (8, 64, 28, 28), (50176, 1, 1792, 64))
    buf63 = buf61; del buf61  # reuse
    cpp_fused_convolution_18(c_void_p(arg57_1.data_ptr()), c_void_p(buf63.data_ptr()))
    del arg57_1
    # Source Nodes: [sp_44], Original ATen: [aten.convolution]
    buf64 = extern_kernels.convolution(reinterpret_tensor(buf60, (8, 64, 56, 56), (802816, 1, 14336, 256), 64), buf63, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf64, (8, 64, 28, 28), (50176, 1, 1792, 64))
    buf65 = buf63; del buf63  # reuse
    cpp_fused_convolution_19(c_void_p(arg60_1.data_ptr()), c_void_p(buf65.data_ptr()))
    del arg60_1
    # Source Nodes: [sp_48], Original ATen: [aten.convolution]
    buf66 = extern_kernels.convolution(reinterpret_tensor(buf60, (8, 64, 56, 56), (802816, 1, 14336, 256), 128), buf65, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf66, (8, 64, 28, 28), (50176, 1, 1792, 64))
    buf71 = reinterpret_tensor(buf4, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf4  # reuse
    buf67 = reinterpret_tensor(buf71, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
    buf68 = reinterpret_tensor(buf71, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
    buf69 = reinterpret_tensor(buf71, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
    buf70 = reinterpret_tensor(buf71, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
    buf72 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_20(c_void_p(buf60.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()))
    del arg311_1
    del arg312_1
    del arg314_1
    del arg315_1
    del arg317_1
    del arg318_1
    del arg55_1
    del arg56_1
    del arg58_1
    del arg59_1
    del arg61_1
    del arg62_1
    del buf60
    del buf62
    del buf64
    del buf67
    del buf68
    del buf69
    del buf70
    # Source Nodes: [out_28], Original ATen: [aten.convolution]
    buf73 = extern_kernels.convolution(buf72, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf73, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del arg63_1
    # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
    buf74 = extern_kernels.convolution(buf58, arg66_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf74, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del arg66_1
    del buf58
    buf75 = buf73; del buf73  # reuse
    buf76 = buf75; del buf75  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_21(c_void_p(buf76.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()))
    del arg320_1
    del arg321_1
    del arg323_1
    del arg324_1
    del arg64_1
    del arg65_1
    del arg67_1
    del arg68_1
    del buf74
    # Source Nodes: [out_32, shortcut_6], Original ATen: [aten.convolution, aten.relu]
    buf77 = extern_kernels.convolution(buf76, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf77, (8, 256, 28, 28), (200704, 1, 7168, 256))
    del arg69_1
    buf78 = buf77; del buf77  # reuse
    buf79 = buf65; del buf65  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_22(c_void_p(buf78.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(buf79.data_ptr()))
    del arg326_1
    del arg327_1
    del arg70_1
    del arg71_1
    del arg72_1
    # Source Nodes: [sp_53], Original ATen: [aten.convolution]
    buf80 = extern_kernels.convolution(reinterpret_tensor(buf78, (8, 64, 28, 28), (200704, 1, 7168, 256), 0), buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf80, (8, 64, 28, 28), (50176, 1, 1792, 64))
    buf91 = reinterpret_tensor(buf72, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf72  # reuse
    buf81 = reinterpret_tensor(buf91, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
    buf82 = buf66; del buf66  # reuse
    buf83 = buf79; del buf79  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_23(c_void_p(buf80.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    del arg329_1
    del arg330_1
    del arg73_1
    del arg74_1
    del arg75_1
    del buf80
    # Source Nodes: [sp_56, sp_57], Original ATen: [aten.add, aten.convolution]
    buf84 = extern_kernels.convolution(buf82, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf84, (8, 64, 28, 28), (50176, 1, 1792, 64))
    buf85 = reinterpret_tensor(buf91, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
    buf86 = buf82; del buf82  # reuse
    buf87 = buf83; del buf83  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_24(c_void_p(buf84.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()))
    del arg332_1
    del arg333_1
    del arg76_1
    del arg77_1
    del arg78_1
    del buf84
    # Source Nodes: [sp_60, sp_61], Original ATen: [aten.add, aten.convolution]
    buf88 = extern_kernels.convolution(buf86, buf87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf88, (8, 64, 28, 28), (50176, 1, 1792, 64))
    del buf86
    buf89 = reinterpret_tensor(buf91, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
    buf90 = reinterpret_tensor(buf91, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
    buf92 = reinterpret_tensor(buf71, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf71  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_25(c_void_p(buf88.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf92.data_ptr()))
    del arg335_1
    del arg336_1
    del arg79_1
    del arg80_1
    del buf78
    del buf81
    del buf85
    del buf89
    del buf90
    # Source Nodes: [out_36], Original ATen: [aten.convolution]
    buf93 = extern_kernels.convolution(buf92, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf93, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del arg81_1
    buf94 = buf76; del buf76  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_26(c_void_p(buf94.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg338_1
    del arg339_1
    del arg82_1
    del arg83_1
    del buf93
    # Source Nodes: [out_40], Original ATen: [aten.convolution]
    buf95 = extern_kernels.convolution(buf94, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf95, (8, 256, 28, 28), (200704, 1, 7168, 256))
    del arg84_1
    buf96 = buf95; del buf95  # reuse
    buf97 = buf87; del buf87  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_27(c_void_p(buf96.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf97.data_ptr()))
    del arg341_1
    del arg342_1
    del arg85_1
    del arg86_1
    del arg87_1
    # Source Nodes: [sp_66], Original ATen: [aten.convolution]
    buf98 = extern_kernels.convolution(reinterpret_tensor(buf96, (8, 64, 28, 28), (200704, 1, 7168, 256), 0), buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf98, (8, 64, 28, 28), (50176, 1, 1792, 64))
    buf109 = reinterpret_tensor(buf92, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf92  # reuse
    buf99 = reinterpret_tensor(buf109, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
    buf100 = buf88; del buf88  # reuse
    buf101 = buf97; del buf97  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_28(c_void_p(buf98.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    del arg344_1
    del arg345_1
    del arg88_1
    del arg89_1
    del arg90_1
    del buf98
    # Source Nodes: [sp_69, sp_70], Original ATen: [aten.add, aten.convolution]
    buf102 = extern_kernels.convolution(buf100, buf101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf102, (8, 64, 28, 28), (50176, 1, 1792, 64))
    buf103 = reinterpret_tensor(buf109, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
    buf104 = buf100; del buf100  # reuse
    buf105 = buf101; del buf101  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_29(c_void_p(buf102.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    del arg347_1
    del arg348_1
    del arg91_1
    del arg92_1
    del arg93_1
    del buf102
    # Source Nodes: [sp_73, sp_74], Original ATen: [aten.add, aten.convolution]
    buf106 = extern_kernels.convolution(buf104, buf105, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf106, (8, 64, 28, 28), (50176, 1, 1792, 64))
    del buf104
    buf107 = reinterpret_tensor(buf109, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
    buf108 = reinterpret_tensor(buf109, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
    buf110 = reinterpret_tensor(buf91, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf91  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_30(c_void_p(buf106.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf110.data_ptr()))
    del arg350_1
    del arg351_1
    del arg94_1
    del arg95_1
    del buf103
    del buf107
    del buf108
    del buf109
    del buf99
    # Source Nodes: [out_44], Original ATen: [aten.convolution]
    buf111 = extern_kernels.convolution(buf110, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf111, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del arg96_1
    buf112 = buf111; del buf111  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_31(c_void_p(buf112.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(buf94.data_ptr()))
    del arg353_1
    del arg354_1
    del arg97_1
    del arg98_1
    del buf94
    # Source Nodes: [out_48], Original ATen: [aten.convolution]
    buf113 = extern_kernels.convolution(buf112, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf113, (8, 256, 28, 28), (200704, 1, 7168, 256))
    del arg99_1
    buf114 = buf113; del buf113  # reuse
    buf115 = buf105; del buf105  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_32(c_void_p(buf114.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(buf115.data_ptr()))
    del arg100_1
    del arg101_1
    del arg102_1
    del arg356_1
    del arg357_1
    # Source Nodes: [sp_79], Original ATen: [aten.convolution]
    buf116 = extern_kernels.convolution(reinterpret_tensor(buf114, (8, 64, 28, 28), (200704, 1, 7168, 256), 0), buf115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf116, (8, 64, 28, 28), (50176, 1, 1792, 64))
    buf127 = reinterpret_tensor(buf110, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf110  # reuse
    buf117 = reinterpret_tensor(buf127, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
    buf118 = buf106; del buf106  # reuse
    buf119 = buf115; del buf115  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_33(c_void_p(buf116.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()))
    del arg103_1
    del arg104_1
    del arg105_1
    del arg359_1
    del arg360_1
    del buf116
    # Source Nodes: [sp_82, sp_83], Original ATen: [aten.add, aten.convolution]
    buf120 = extern_kernels.convolution(buf118, buf119, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf120, (8, 64, 28, 28), (50176, 1, 1792, 64))
    buf121 = reinterpret_tensor(buf127, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
    buf122 = buf118; del buf118  # reuse
    buf123 = buf119; del buf119  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_34(c_void_p(buf120.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg363_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    del arg106_1
    del arg107_1
    del arg108_1
    del arg362_1
    del arg363_1
    del buf120
    # Source Nodes: [sp_86, sp_87], Original ATen: [aten.add, aten.convolution]
    buf124 = extern_kernels.convolution(buf122, buf123, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf124, (8, 64, 28, 28), (50176, 1, 1792, 64))
    del buf123
    buf125 = reinterpret_tensor(buf127, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
    buf126 = reinterpret_tensor(buf127, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
    buf128 = buf96; del buf96  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_35(c_void_p(buf124.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()))
    del arg109_1
    del arg110_1
    del arg365_1
    del arg366_1
    del buf114
    del buf117
    del buf121
    del buf125
    del buf126
    del buf127
    # Source Nodes: [out_52], Original ATen: [aten.convolution]
    buf129 = extern_kernels.convolution(buf128, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf129, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del arg111_1
    del buf128
    buf130 = buf112; del buf112  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_36(c_void_p(buf130.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()))
    del arg112_1
    del arg113_1
    del arg368_1
    del arg369_1
    del buf129
    # Source Nodes: [out_56], Original ATen: [aten.convolution]
    buf131 = extern_kernels.convolution(buf130, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf131, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del arg114_1
    buf132 = buf131; del buf131  # reuse
    buf133 = empty_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_37(c_void_p(buf132.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(buf133.data_ptr()))
    del arg115_1
    del arg116_1
    del arg117_1
    del arg371_1
    del arg372_1
    # Source Nodes: [sp_92], Original ATen: [aten.convolution]
    buf134 = extern_kernels.convolution(reinterpret_tensor(buf132, (8, 128, 28, 28), (401408, 1, 14336, 512), 0), buf133, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf134, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf135 = buf133; del buf133  # reuse
    cpp_fused_convolution_38(c_void_p(arg120_1.data_ptr()), c_void_p(buf135.data_ptr()))
    del arg120_1
    # Source Nodes: [sp_96], Original ATen: [aten.convolution]
    buf136 = extern_kernels.convolution(reinterpret_tensor(buf132, (8, 128, 28, 28), (401408, 1, 14336, 512), 128), buf135, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf136, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf137 = buf135; del buf135  # reuse
    cpp_fused_convolution_39(c_void_p(arg123_1.data_ptr()), c_void_p(buf137.data_ptr()))
    del arg123_1
    # Source Nodes: [sp_100], Original ATen: [aten.convolution]
    buf138 = extern_kernels.convolution(reinterpret_tensor(buf132, (8, 128, 28, 28), (401408, 1, 14336, 512), 256), buf137, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf138, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf143 = reinterpret_tensor(buf52, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf52  # reuse
    buf139 = reinterpret_tensor(buf143, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
    buf140 = reinterpret_tensor(buf143, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
    buf141 = reinterpret_tensor(buf143, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
    buf142 = reinterpret_tensor(buf143, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
    buf144 = reinterpret_tensor(buf50, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf50  # reuse
    cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_40(c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(arg374_1.data_ptr()), c_void_p(arg375_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(arg377_1.data_ptr()), c_void_p(arg378_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg381_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()))
    del arg118_1
    del arg119_1
    del arg121_1
    del arg122_1
    del arg124_1
    del arg125_1
    del arg374_1
    del arg375_1
    del arg377_1
    del arg378_1
    del arg380_1
    del arg381_1
    del buf132
    del buf134
    del buf136
    del buf139
    del buf140
    del buf141
    del buf142
    # Source Nodes: [out_60], Original ATen: [aten.convolution]
    buf145 = extern_kernels.convolution(buf144, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf145, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg126_1
    # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
    buf146 = extern_kernels.convolution(buf130, arg129_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf146, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg129_1
    del buf130
    buf147 = buf145; del buf145  # reuse
    buf148 = buf147; del buf147  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_41(c_void_p(buf148.data_ptr()), c_void_p(arg383_1.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg387_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()))
    del arg127_1
    del arg128_1
    del arg130_1
    del arg131_1
    del arg383_1
    del arg384_1
    del arg386_1
    del arg387_1
    del buf146
    # Source Nodes: [out_64, shortcut_11], Original ATen: [aten.convolution, aten.relu]
    buf149 = extern_kernels.convolution(buf148, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf149, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del arg132_1
    buf150 = buf149; del buf149  # reuse
    buf151 = buf137; del buf137  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_42(c_void_p(buf150.data_ptr()), c_void_p(arg389_1.data_ptr()), c_void_p(arg390_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(buf151.data_ptr()))
    del arg133_1
    del arg134_1
    del arg135_1
    del arg389_1
    del arg390_1
    # Source Nodes: [sp_105], Original ATen: [aten.convolution]
    buf152 = extern_kernels.convolution(reinterpret_tensor(buf150, (8, 128, 14, 14), (100352, 1, 7168, 512), 0), buf151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf152, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf163 = reinterpret_tensor(buf144, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf144  # reuse
    buf153 = reinterpret_tensor(buf163, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
    buf154 = buf138; del buf138  # reuse
    buf155 = buf151; del buf151  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_43(c_void_p(buf152.data_ptr()), c_void_p(arg392_1.data_ptr()), c_void_p(arg393_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()))
    del arg136_1
    del arg137_1
    del arg138_1
    del arg392_1
    del arg393_1
    del buf152
    # Source Nodes: [sp_108, sp_109], Original ATen: [aten.add, aten.convolution]
    buf156 = extern_kernels.convolution(buf154, buf155, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf156, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf157 = reinterpret_tensor(buf163, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
    buf158 = buf154; del buf154  # reuse
    buf159 = buf155; del buf155  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_44(c_void_p(buf156.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()))
    del arg139_1
    del arg140_1
    del arg141_1
    del arg395_1
    del arg396_1
    del buf156
    # Source Nodes: [sp_112, sp_113], Original ATen: [aten.add, aten.convolution]
    buf160 = extern_kernels.convolution(buf158, buf159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf160, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del buf158
    buf161 = reinterpret_tensor(buf163, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
    buf162 = reinterpret_tensor(buf163, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
    buf164 = reinterpret_tensor(buf143, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf143  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_45(c_void_p(buf160.data_ptr()), c_void_p(arg398_1.data_ptr()), c_void_p(arg399_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()))
    del arg142_1
    del arg143_1
    del arg398_1
    del arg399_1
    del buf150
    del buf153
    del buf157
    del buf161
    del buf162
    # Source Nodes: [out_68], Original ATen: [aten.convolution]
    buf165 = extern_kernels.convolution(buf164, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf165, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg144_1
    buf166 = buf148; del buf148  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_46(c_void_p(buf166.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(arg401_1.data_ptr()), c_void_p(arg402_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()))
    del arg145_1
    del arg146_1
    del arg401_1
    del arg402_1
    del buf165
    # Source Nodes: [out_72], Original ATen: [aten.convolution]
    buf167 = extern_kernels.convolution(buf166, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf167, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del arg147_1
    buf168 = buf167; del buf167  # reuse
    buf169 = buf159; del buf159  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_47(c_void_p(buf168.data_ptr()), c_void_p(arg404_1.data_ptr()), c_void_p(arg405_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(buf169.data_ptr()))
    del arg148_1
    del arg149_1
    del arg150_1
    del arg404_1
    del arg405_1
    # Source Nodes: [sp_118], Original ATen: [aten.convolution]
    buf170 = extern_kernels.convolution(reinterpret_tensor(buf168, (8, 128, 14, 14), (100352, 1, 7168, 512), 0), buf169, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf170, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf181 = reinterpret_tensor(buf164, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf164  # reuse
    buf171 = reinterpret_tensor(buf181, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
    buf172 = buf160; del buf160  # reuse
    buf173 = buf169; del buf169  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_48(c_void_p(buf170.data_ptr()), c_void_p(arg407_1.data_ptr()), c_void_p(arg408_1.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    del arg151_1
    del arg152_1
    del arg153_1
    del arg407_1
    del arg408_1
    del buf170
    # Source Nodes: [sp_121, sp_122], Original ATen: [aten.add, aten.convolution]
    buf174 = extern_kernels.convolution(buf172, buf173, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf174, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf175 = reinterpret_tensor(buf181, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
    buf176 = buf172; del buf172  # reuse
    buf177 = buf173; del buf173  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_49(c_void_p(buf174.data_ptr()), c_void_p(arg410_1.data_ptr()), c_void_p(arg411_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    del arg154_1
    del arg155_1
    del arg156_1
    del arg410_1
    del arg411_1
    del buf174
    # Source Nodes: [sp_125, sp_126], Original ATen: [aten.add, aten.convolution]
    buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf178, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del buf176
    buf179 = reinterpret_tensor(buf181, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
    buf180 = reinterpret_tensor(buf181, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
    buf182 = reinterpret_tensor(buf163, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf163  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_50(c_void_p(buf178.data_ptr()), c_void_p(arg413_1.data_ptr()), c_void_p(arg414_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()))
    del arg157_1
    del arg158_1
    del arg413_1
    del arg414_1
    del buf168
    del buf171
    del buf175
    del buf179
    del buf180
    # Source Nodes: [out_76], Original ATen: [aten.convolution]
    buf183 = extern_kernels.convolution(buf182, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf183, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg159_1
    buf184 = buf166; del buf166  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_51(c_void_p(buf184.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(arg416_1.data_ptr()), c_void_p(arg417_1.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(arg161_1.data_ptr()))
    del arg160_1
    del arg161_1
    del arg416_1
    del arg417_1
    del buf183
    # Source Nodes: [out_80], Original ATen: [aten.convolution]
    buf185 = extern_kernels.convolution(buf184, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf185, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del arg162_1
    buf186 = buf185; del buf185  # reuse
    buf187 = buf177; del buf177  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_52(c_void_p(buf186.data_ptr()), c_void_p(arg419_1.data_ptr()), c_void_p(arg420_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(buf187.data_ptr()))
    del arg163_1
    del arg164_1
    del arg165_1
    del arg419_1
    del arg420_1
    # Source Nodes: [sp_131], Original ATen: [aten.convolution]
    buf188 = extern_kernels.convolution(reinterpret_tensor(buf186, (8, 128, 14, 14), (100352, 1, 7168, 512), 0), buf187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf188, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf199 = reinterpret_tensor(buf182, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf182  # reuse
    buf189 = reinterpret_tensor(buf199, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
    buf190 = buf178; del buf178  # reuse
    buf191 = buf187; del buf187  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_53(c_void_p(buf188.data_ptr()), c_void_p(arg422_1.data_ptr()), c_void_p(arg423_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    del arg166_1
    del arg167_1
    del arg168_1
    del arg422_1
    del arg423_1
    del buf188
    # Source Nodes: [sp_134, sp_135], Original ATen: [aten.add, aten.convolution]
    buf192 = extern_kernels.convolution(buf190, buf191, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf192, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf193 = reinterpret_tensor(buf199, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
    buf194 = buf190; del buf190  # reuse
    buf195 = buf191; del buf191  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_54(c_void_p(buf192.data_ptr()), c_void_p(arg425_1.data_ptr()), c_void_p(arg426_1.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    del arg169_1
    del arg170_1
    del arg171_1
    del arg425_1
    del arg426_1
    del buf192
    # Source Nodes: [sp_138, sp_139], Original ATen: [aten.add, aten.convolution]
    buf196 = extern_kernels.convolution(buf194, buf195, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf196, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del buf194
    buf197 = reinterpret_tensor(buf199, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
    buf198 = reinterpret_tensor(buf199, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
    buf200 = reinterpret_tensor(buf181, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf181  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_55(c_void_p(buf196.data_ptr()), c_void_p(arg428_1.data_ptr()), c_void_p(arg429_1.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf200.data_ptr()))
    del arg172_1
    del arg173_1
    del arg428_1
    del arg429_1
    del buf186
    del buf189
    del buf193
    del buf197
    del buf198
    # Source Nodes: [out_84], Original ATen: [aten.convolution]
    buf201 = extern_kernels.convolution(buf200, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf201, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg174_1
    buf202 = buf184; del buf184  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_56(c_void_p(buf202.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(arg431_1.data_ptr()), c_void_p(arg432_1.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(arg176_1.data_ptr()))
    del arg175_1
    del arg176_1
    del arg431_1
    del arg432_1
    del buf201
    # Source Nodes: [out_88], Original ATen: [aten.convolution]
    buf203 = extern_kernels.convolution(buf202, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf203, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del arg177_1
    buf204 = buf203; del buf203  # reuse
    buf205 = buf195; del buf195  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_57(c_void_p(buf204.data_ptr()), c_void_p(arg434_1.data_ptr()), c_void_p(arg435_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(buf205.data_ptr()))
    del arg178_1
    del arg179_1
    del arg180_1
    del arg434_1
    del arg435_1
    # Source Nodes: [sp_144], Original ATen: [aten.convolution]
    buf206 = extern_kernels.convolution(reinterpret_tensor(buf204, (8, 128, 14, 14), (100352, 1, 7168, 512), 0), buf205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf206, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf217 = reinterpret_tensor(buf200, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf200  # reuse
    buf207 = reinterpret_tensor(buf217, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
    buf208 = buf196; del buf196  # reuse
    buf209 = buf205; del buf205  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_58(c_void_p(buf206.data_ptr()), c_void_p(arg437_1.data_ptr()), c_void_p(arg438_1.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()))
    del arg181_1
    del arg182_1
    del arg183_1
    del arg437_1
    del arg438_1
    del buf206
    # Source Nodes: [sp_147, sp_148], Original ATen: [aten.add, aten.convolution]
    buf210 = extern_kernels.convolution(buf208, buf209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf210, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf211 = reinterpret_tensor(buf217, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
    buf212 = buf208; del buf208  # reuse
    buf213 = buf209; del buf209  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_59(c_void_p(buf210.data_ptr()), c_void_p(arg440_1.data_ptr()), c_void_p(arg441_1.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    del arg184_1
    del arg185_1
    del arg186_1
    del arg440_1
    del arg441_1
    del buf210
    # Source Nodes: [sp_151, sp_152], Original ATen: [aten.add, aten.convolution]
    buf214 = extern_kernels.convolution(buf212, buf213, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf214, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del buf212
    buf215 = reinterpret_tensor(buf217, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
    buf216 = reinterpret_tensor(buf217, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
    buf218 = reinterpret_tensor(buf199, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf199  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_60(c_void_p(buf214.data_ptr()), c_void_p(arg443_1.data_ptr()), c_void_p(arg444_1.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf218.data_ptr()))
    del arg187_1
    del arg188_1
    del arg443_1
    del arg444_1
    del buf204
    del buf207
    del buf211
    del buf215
    del buf216
    # Source Nodes: [out_92], Original ATen: [aten.convolution]
    buf219 = extern_kernels.convolution(buf218, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf219, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg189_1
    buf220 = buf202; del buf202  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_61(c_void_p(buf220.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(arg446_1.data_ptr()), c_void_p(arg447_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg191_1.data_ptr()))
    del arg190_1
    del arg191_1
    del arg446_1
    del arg447_1
    del buf219
    # Source Nodes: [out_96], Original ATen: [aten.convolution]
    buf221 = extern_kernels.convolution(buf220, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf221, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del arg192_1
    buf222 = buf221; del buf221  # reuse
    buf223 = buf213; del buf213  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_62(c_void_p(buf222.data_ptr()), c_void_p(arg449_1.data_ptr()), c_void_p(arg450_1.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(buf223.data_ptr()))
    del arg193_1
    del arg194_1
    del arg195_1
    del arg449_1
    del arg450_1
    # Source Nodes: [sp_157], Original ATen: [aten.convolution]
    buf224 = extern_kernels.convolution(reinterpret_tensor(buf222, (8, 128, 14, 14), (100352, 1, 7168, 512), 0), buf223, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf224, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf235 = reinterpret_tensor(buf218, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf218  # reuse
    buf225 = reinterpret_tensor(buf235, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
    buf226 = buf214; del buf214  # reuse
    buf227 = buf223; del buf223  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_63(c_void_p(buf224.data_ptr()), c_void_p(arg452_1.data_ptr()), c_void_p(arg453_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()))
    del arg196_1
    del arg197_1
    del arg198_1
    del arg452_1
    del arg453_1
    del buf224
    # Source Nodes: [sp_160, sp_161], Original ATen: [aten.add, aten.convolution]
    buf228 = extern_kernels.convolution(buf226, buf227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf228, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf229 = reinterpret_tensor(buf235, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
    buf230 = buf226; del buf226  # reuse
    buf231 = buf227; del buf227  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_64(c_void_p(buf228.data_ptr()), c_void_p(arg455_1.data_ptr()), c_void_p(arg456_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del arg199_1
    del arg200_1
    del arg201_1
    del arg455_1
    del arg456_1
    del buf228
    # Source Nodes: [sp_164, sp_165], Original ATen: [aten.add, aten.convolution]
    buf232 = extern_kernels.convolution(buf230, buf231, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf232, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del buf230
    del buf231
    buf233 = reinterpret_tensor(buf235, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
    buf234 = reinterpret_tensor(buf235, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
    buf236 = reinterpret_tensor(buf217, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf217  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_65(c_void_p(buf232.data_ptr()), c_void_p(arg458_1.data_ptr()), c_void_p(arg459_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf236.data_ptr()))
    del arg202_1
    del arg203_1
    del arg458_1
    del arg459_1
    del buf222
    del buf225
    del buf229
    del buf232
    del buf233
    del buf234
    del buf235
    # Source Nodes: [out_100], Original ATen: [aten.convolution]
    buf237 = extern_kernels.convolution(buf236, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf237, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg204_1
    del buf236
    buf238 = buf220; del buf220  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_66(c_void_p(buf238.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(arg461_1.data_ptr()), c_void_p(arg462_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg206_1.data_ptr()))
    del arg205_1
    del arg206_1
    del arg461_1
    del arg462_1
    del buf237
    # Source Nodes: [out_104], Original ATen: [aten.convolution]
    buf239 = extern_kernels.convolution(buf238, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf239, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg207_1
    buf240 = buf239; del buf239  # reuse
    buf241 = empty_strided((256, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_67(c_void_p(buf240.data_ptr()), c_void_p(arg464_1.data_ptr()), c_void_p(arg465_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(buf241.data_ptr()))
    del arg208_1
    del arg209_1
    del arg210_1
    del arg464_1
    del arg465_1
    # Source Nodes: [sp_170], Original ATen: [aten.convolution]
    buf242 = extern_kernels.convolution(reinterpret_tensor(buf240, (8, 256, 14, 14), (200704, 1, 14336, 1024), 0), buf241, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf242, (8, 256, 7, 7), (12544, 1, 1792, 256))
    buf243 = buf241; del buf241  # reuse
    cpp_fused_convolution_68(c_void_p(arg213_1.data_ptr()), c_void_p(buf243.data_ptr()))
    del arg213_1
    # Source Nodes: [sp_174], Original ATen: [aten.convolution]
    buf244 = extern_kernels.convolution(reinterpret_tensor(buf240, (8, 256, 14, 14), (200704, 1, 14336, 1024), 256), buf243, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf244, (8, 256, 7, 7), (12544, 1, 1792, 256))
    buf245 = buf243; del buf243  # reuse
    cpp_fused_convolution_69(c_void_p(arg216_1.data_ptr()), c_void_p(buf245.data_ptr()))
    del arg216_1
    # Source Nodes: [sp_178], Original ATen: [aten.convolution]
    buf246 = extern_kernels.convolution(reinterpret_tensor(buf240, (8, 256, 14, 14), (200704, 1, 14336, 1024), 512), buf245, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf246, (8, 256, 7, 7), (12544, 1, 1792, 256))
    buf251 = reinterpret_tensor(buf124, (8, 1024, 7, 7), (50176, 49, 7, 1), 0); del buf124  # reuse
    buf247 = reinterpret_tensor(buf251, (8, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
    buf248 = reinterpret_tensor(buf251, (8, 256, 7, 7), (50176, 49, 7, 1), 0)  # alias
    buf249 = reinterpret_tensor(buf251, (8, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
    buf250 = reinterpret_tensor(buf251, (8, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
    buf252 = reinterpret_tensor(buf122, (8, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf122  # reuse
    cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_70(c_void_p(buf240.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(arg467_1.data_ptr()), c_void_p(arg468_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(arg470_1.data_ptr()), c_void_p(arg471_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(arg473_1.data_ptr()), c_void_p(arg474_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf252.data_ptr()))
    del arg211_1
    del arg212_1
    del arg214_1
    del arg215_1
    del arg217_1
    del arg218_1
    del arg467_1
    del arg468_1
    del arg470_1
    del arg471_1
    del arg473_1
    del arg474_1
    del buf240
    del buf242
    del buf244
    del buf247
    del buf248
    del buf249
    del buf250
    # Source Nodes: [out_108], Original ATen: [aten.convolution]
    buf253 = extern_kernels.convolution(buf252, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf253, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    del arg219_1
    # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
    buf254 = extern_kernels.convolution(buf238, arg222_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf254, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    del arg222_1
    del buf238
    buf255 = buf253; del buf253  # reuse
    buf256 = buf255; del buf255  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_71(c_void_p(buf256.data_ptr()), c_void_p(arg476_1.data_ptr()), c_void_p(arg477_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(arg479_1.data_ptr()), c_void_p(arg480_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg224_1.data_ptr()))
    del arg220_1
    del arg221_1
    del arg223_1
    del arg224_1
    del arg476_1
    del arg477_1
    del arg479_1
    del arg480_1
    del buf254
    # Source Nodes: [out_112, shortcut_18], Original ATen: [aten.convolution, aten.relu]
    buf257 = extern_kernels.convolution(buf256, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf257, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    del arg225_1
    buf258 = buf257; del buf257  # reuse
    buf259 = buf245; del buf245  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_72(c_void_p(buf258.data_ptr()), c_void_p(arg482_1.data_ptr()), c_void_p(arg483_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(buf259.data_ptr()))
    del arg226_1
    del arg227_1
    del arg228_1
    del arg482_1
    del arg483_1
    # Source Nodes: [sp_183], Original ATen: [aten.convolution]
    buf260 = extern_kernels.convolution(reinterpret_tensor(buf258, (8, 256, 7, 7), (50176, 1, 7168, 1024), 0), buf259, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf260, (8, 256, 7, 7), (12544, 1, 1792, 256))
    buf271 = reinterpret_tensor(buf252, (8, 1024, 7, 7), (50176, 49, 7, 1), 0); del buf252  # reuse
    buf261 = reinterpret_tensor(buf271, (8, 256, 7, 7), (50176, 49, 7, 1), 0)  # alias
    buf262 = buf246; del buf246  # reuse
    buf263 = buf259; del buf259  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_73(c_void_p(buf260.data_ptr()), c_void_p(arg485_1.data_ptr()), c_void_p(arg486_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()))
    del arg229_1
    del arg230_1
    del arg231_1
    del arg485_1
    del arg486_1
    del buf260
    # Source Nodes: [sp_186, sp_187], Original ATen: [aten.add, aten.convolution]
    buf264 = extern_kernels.convolution(buf262, buf263, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf264, (8, 256, 7, 7), (12544, 1, 1792, 256))
    buf265 = reinterpret_tensor(buf271, (8, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
    buf266 = buf262; del buf262  # reuse
    buf267 = buf263; del buf263  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_74(c_void_p(buf264.data_ptr()), c_void_p(arg488_1.data_ptr()), c_void_p(arg489_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()))
    del arg232_1
    del arg233_1
    del arg234_1
    del arg488_1
    del arg489_1
    del buf264
    # Source Nodes: [sp_190, sp_191], Original ATen: [aten.add, aten.convolution]
    buf268 = extern_kernels.convolution(buf266, buf267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf268, (8, 256, 7, 7), (12544, 1, 1792, 256))
    del buf266
    buf269 = reinterpret_tensor(buf271, (8, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
    buf270 = reinterpret_tensor(buf271, (8, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
    buf272 = reinterpret_tensor(buf251, (8, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf251  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_75(c_void_p(buf268.data_ptr()), c_void_p(arg491_1.data_ptr()), c_void_p(arg492_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf272.data_ptr()))
    del arg235_1
    del arg236_1
    del arg491_1
    del arg492_1
    del buf258
    del buf261
    del buf265
    del buf269
    del buf270
    # Source Nodes: [out_116], Original ATen: [aten.convolution]
    buf273 = extern_kernels.convolution(buf272, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf273, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    del arg237_1
    buf274 = buf256; del buf256  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_76(c_void_p(buf274.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(arg494_1.data_ptr()), c_void_p(arg495_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg239_1.data_ptr()))
    del arg238_1
    del arg239_1
    del arg494_1
    del arg495_1
    del buf273
    # Source Nodes: [out_120], Original ATen: [aten.convolution]
    buf275 = extern_kernels.convolution(buf274, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf275, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    del arg240_1
    buf276 = buf275; del buf275  # reuse
    buf277 = buf267; del buf267  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_77(c_void_p(buf276.data_ptr()), c_void_p(arg497_1.data_ptr()), c_void_p(arg498_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(buf277.data_ptr()))
    del arg241_1
    del arg242_1
    del arg243_1
    del arg497_1
    del arg498_1
    # Source Nodes: [sp_196], Original ATen: [aten.convolution]
    buf278 = extern_kernels.convolution(reinterpret_tensor(buf276, (8, 256, 7, 7), (50176, 1, 7168, 1024), 0), buf277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf278, (8, 256, 7, 7), (12544, 1, 1792, 256))
    buf289 = reinterpret_tensor(buf272, (8, 1024, 7, 7), (50176, 49, 7, 1), 0); del buf272  # reuse
    buf279 = reinterpret_tensor(buf289, (8, 256, 7, 7), (50176, 49, 7, 1), 0)  # alias
    buf280 = buf268; del buf268  # reuse
    buf281 = buf277; del buf277  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_78(c_void_p(buf278.data_ptr()), c_void_p(arg500_1.data_ptr()), c_void_p(arg501_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    del arg244_1
    del arg245_1
    del arg246_1
    del arg500_1
    del arg501_1
    del buf278
    # Source Nodes: [sp_199, sp_200], Original ATen: [aten.add, aten.convolution]
    buf282 = extern_kernels.convolution(buf280, buf281, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf282, (8, 256, 7, 7), (12544, 1, 1792, 256))
    buf283 = reinterpret_tensor(buf289, (8, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
    buf284 = buf280; del buf280  # reuse
    buf285 = buf281; del buf281  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_79(c_void_p(buf282.data_ptr()), c_void_p(arg503_1.data_ptr()), c_void_p(arg504_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    del arg247_1
    del arg248_1
    del arg249_1
    del arg503_1
    del arg504_1
    del buf282
    # Source Nodes: [sp_203, sp_204], Original ATen: [aten.add, aten.convolution]
    buf286 = extern_kernels.convolution(buf284, buf285, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf286, (8, 256, 7, 7), (12544, 1, 1792, 256))
    del buf284
    del buf285
    buf287 = reinterpret_tensor(buf289, (8, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
    buf288 = reinterpret_tensor(buf289, (8, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
    buf290 = reinterpret_tensor(buf271, (8, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf271  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_relu_80(c_void_p(buf286.data_ptr()), c_void_p(arg506_1.data_ptr()), c_void_p(arg507_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf290.data_ptr()))
    del arg250_1
    del arg251_1
    del arg506_1
    del arg507_1
    del buf276
    del buf279
    del buf283
    del buf286
    del buf287
    del buf288
    del buf289
    # Source Nodes: [out_124], Original ATen: [aten.convolution]
    buf291 = extern_kernels.convolution(buf290, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf291, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    del arg252_1
    del buf290
    buf292 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cpu', dtype=torch.float32)
    buf293 = reinterpret_tensor(buf292, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf292  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_81(c_void_p(buf293.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(arg509_1.data_ptr()), c_void_p(arg510_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(buf274.data_ptr()))
    del arg253_1
    del arg254_1
    del arg509_1
    del arg510_1
    del buf274
    del buf291
    buf294 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg256_1, reinterpret_tensor(buf293, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg255_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf294)
    del arg255_1
    del arg256_1
    return (buf294, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg260_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg263_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg266_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg269_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg272_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg275_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg278_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg281_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg284_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg287_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg290_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg293_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg296_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg299_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg302_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg305_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg308_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg311_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg314_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg317_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg320_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg323_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg326_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg329_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg332_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg335_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg338_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg341_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg344_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg347_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg350_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg353_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg356_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg359_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg362_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg365_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg368_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg371_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg374_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg377_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg380_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg383_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg386_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg389_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg392_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg395_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg397_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg398_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg400_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg401_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg403_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg404_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg406_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg407_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg409_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg410_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg412_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg413_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg415_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg416_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg418_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg419_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg421_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg422_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg424_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg425_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg427_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg428_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg430_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg431_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg433_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg434_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg436_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg437_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg439_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg440_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg442_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg443_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg445_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg446_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg448_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg449_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg451_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg452_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg454_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg455_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg457_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg458_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg459_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg460_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg461_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg462_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg463_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg464_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg465_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg466_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg467_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg468_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg469_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg470_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg471_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg472_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg473_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg474_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg475_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg476_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg477_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg478_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg479_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg480_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg481_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg482_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg483_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg484_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg485_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg486_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg487_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg488_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg489_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg490_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg491_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg492_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg493_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg494_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg495_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg496_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg497_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg498_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg499_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg500_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg501_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg502_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg503_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg504_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg505_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg506_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg507_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg508_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg509_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg510_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg511_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg512_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2next50', benchmark_compiled_module)
